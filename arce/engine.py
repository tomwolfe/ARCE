import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
import optax
from .models import MSVIB
from .metrics import (
    information_bottleneck_loss, 
    effective_information_robust, 
    estimate_transition_matrix, 
    causal_emergence_loss,
    kl_divergence
)
import jraph
from .utils import laplacian_eigen_spectrum, suggest_num_clusters, BasisLibrary

class ARCE:
    def __init__(self, config):
        self.config = config
        self.model = MSVIB(
            encoder_features=config.get('encoder_features', [64, 64]),
            latent_dim=config.get('latent_dim', 16),
            num_clusters_list=config.get('num_clusters_list', [4]),
            top_k_edges=config.get('top_k_edges', None)
        )
        self.params = None
        self.state = None
        # Pre-calculated Micro-EI for Ising Model (approximate baseline)
        self.micro_ei_baseline = 0.5 
        self.lambda_sparse = config.get('lambda_sparse', 0.01)
        self.basis_library = BasisLibrary(
            include_transcendental=config.get('include_transcendental', True),
            include_quadratic=config.get('include_quadratic', True)
        )

    def init_model(self, rng, sample_graph):
        variables = self.model.init(rng, sample_graph)
        params = variables['params']
        
        # Add learnable loss weights (log variances for uncertainty weighting)
        params['loss_logvars'] = {
            'task': jnp.zeros(()),
            'ce': jnp.zeros(()),
            'continuity': jnp.zeros(()),
            'sparsity': jnp.zeros(()),
            'symbolic': jnp.zeros(()),
            'reconstruction': jnp.zeros(()),
            'predictive': jnp.zeros(()),
            'topo_sparsity': jnp.zeros(())
        }

        # Initialize global symbolic coefficients (Recommendation 1)
        d = self.config.get('latent_dim', 16)
        n_basis = len(self.basis_library.get_names(d))
        params['symbolic_coeffs'] = jnp.zeros((n_basis, d))
        
        self.params = params
        tx = optax.adam(learning_rate=self.config.get('lr', 1e-3))
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=tx
        )

    def analyze_graph_spectrum(self, sample_graph):
        """Analyze the Laplacian Eigen-spectrum to suggest optimal cluster count."""
        eigs = laplacian_eigen_spectrum(sample_graph)
        n_suggested = suggest_num_clusters(eigs)
        print(f"ARCE Spectrum Analysis: Found eigengap suggesting {n_suggested} macro-nodes.")
        return n_suggested

    def coarsen(self, micro_data: jraph.GraphsTuple, rng=None):
        """Returns optimal macro-variables."""
        rngs = {}
        if rng is not None:
            rng1, rng2 = jax.random.split(rng)
            rngs = {'vmap_rng': rng1, 'gumbel': rng2}
        
        # Filter out loss_logvars and symbolic_coeffs from params for model call
        model_params = {k: v for k, v in self.state.params.items() if k not in ['loss_logvars', 'symbolic_coeffs']}
        mu, logvar, pred_y, assignments, coarse_graph, recon_micro, h_micro, all_coarse_adjs = self.model.apply(
            {'params': model_params}, 
            micro_data,
            training=False,
            rngs=rngs
        )
        return coarse_graph, mu

    @staticmethod
    def _lasso_ista_jax(X_mat, y_mat, alpha=0.01, num_iters=100, initial_w=None):
        """
        Pure JAX ISTA for differentiable symbolic discovery.
        Uses jax.lax.scan to allow gradients to flow through iterations.
        """
        def soft_threshold(x, thresh):
            return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0)

        n_feat = X_mat.shape[1]
        n_target = y_mat.shape[1]
        
        spectral_norm_sq = jnp.linalg.norm(X_mat, ord=2)**2
        step = 0.5 / (spectral_norm_sq + 1e-5)
        
        if initial_w is None:
            initial_w = jnp.zeros((n_feat, n_target))

        def scan_body(w, _):
            grad = X_mat.T @ (X_mat @ w - y_mat)
            next_w = soft_threshold(w - step * grad, alpha * step)
            return next_w, None

        w_final, _ = jax.lax.scan(scan_body, initial_w, jnp.arange(num_iters))
        return w_final

    def discover_dynamics(self, time_series_graphs):
        """
        Identifies sparse dynamics in the coarse-grained latent space.
        """
        print("ARCE: Identifying Sufficient Statistics...")
        
        # 1. Extract macro-states over time
        macro_states = []
        for g in time_series_graphs:
            _, mu = self.coarsen(g)
            macro_states.append(mu)
        
        macro_states = jnp.concatenate(macro_states, axis=0) # [time, latent_dim]
        d = macro_states.shape[1]
        
        # 2. Use the learned global coefficients
        coeffs = self.state.params['symbolic_coeffs']
        names = self.basis_library.get_names(d)
        
        # 3. Format the discovered equation
        equations = []
        for i in range(d):
            c_target = coeffs[:, i]
            terms = [f"{float(c):.3f}*{name}" for c, name in zip(c_target, names) if abs(c) > 1e-4]
            eq = " + ".join(terms) if terms else "0"
            equations.append(f"dM{i}/dt = {eq}")
            
        return " | ".join(equations)

    def predict_manifold(self, state_mu, horizon=10):
        """
        Returns probabilistic future state distribution.
        """
        print(f"ARCE: Predicting manifold over horizon {horizon}...")
        
        current_state = state_mu
        path = [current_state]
        
        for _ in range(horizon):
            # For demo, simple drift
            current_state = current_state * 0.95 + 0.01 
            path.append(current_state)
            
        return {
            "mean": current_state,
            "path": jnp.concatenate(path, axis=0),
            "uncertainty": jnp.ones_like(current_state) * (0.05 * horizon)
        }

    def train_step(self, state, batch_graphs, targets, rng):
        # We define a pure function for JIT
        def loss_fn(params, graphs, y, r, micro_ei):
            # Extract logvars for dynamic weighting
            logvars = params['loss_logvars']
            global_coeffs = params['symbolic_coeffs']
            model_params = {k: v for k, v in params.items() if k not in ['loss_logvars', 'symbolic_coeffs']}
            
            r1, r2 = jax.random.split(r)
            mu, logvar, pred_y, _, _, recon_micro, h_micro, all_coarse_adjs = self.model.apply(
                {'params': model_params}, 
                graphs,
                training=True,
                rngs={'vmap_rng': r1, 'gumbel': r2}
            )
            
            # 1. Causal Emergence Loss (Robust) with MI constraint
            macro_ei = effective_information_robust(
                mu, 
                use_gaussian=self.config.get('use_gaussian_ei', False)
            )
            
            # Approximate MI between micro and macro (I(X;M) = H(M) - H(M|X))
            h_m_given_x = 0.5 * jnp.sum(logvar + jnp.log(2 * jnp.pi * jnp.e), axis=-1).mean()
            cov_mu = jnp.cov(mu, rowvar=False)
            _, logdet_cov = jnp.linalg.slogdet(cov_mu + 1e-6 * jnp.eye(mu.shape[1]))
            h_m = 0.5 * (mu.shape[1] * jnp.log(2 * jnp.pi * jnp.e) + logdet_cov)
            mi_micro_macro = h_m - h_m_given_x
            
            ce_loss = causal_emergence_loss(micro_ei, macro_ei, mi_micro_macro, target_mi=self.config.get('target_mi', 1.0))
            
            # 2. Dynamics Consistency (Self-Supervised)
            z_t = mu[:-1]
            z_next = mu[1:]
            def huber_loss(x, delta=1.0):
                abs_x = jnp.abs(x)
                quadratic = jnp.minimum(abs_x, delta)
                linear = abs_x - quadratic
                return 0.5 * jnp.square(quadratic) + delta * linear
            
            continuity_loss = jnp.mean(huber_loss(z_next - z_t))
            
            # 3. Task Loss
            is_unsupervised = (y.shape[0] == 0)
            def supervised_loss():
                return information_bottleneck_loss(mu[-1:], logvar[-1:], pred_y[-1:], y)
            def unsupervised_loss():
                return 0.1 * jnp.mean(kl_divergence(mu, logvar))
            task_loss = jax.lax.cond(is_unsupervised, unsupervised_loss, supervised_loss)
            
            # 4. Sparsity Regularization
            delta_mu = mu[1:] - mu[:-1]
            sparsity_loss = jnp.mean(jnp.abs(delta_mu))

            # 5. Symbolic Discovery Feedback (End-to-End)
            X_poly = self.basis_library.get_features(z_t)
            y_dot = z_next - z_t
            
            # Symbolic residual loss to couple SGD with the dynamics
            symbolic_residual = y_dot - X_poly @ global_coeffs
            symbolic_loss = jnp.mean(jnp.square(symbolic_residual))

            # 6. Reconstruction Loss (VAE term)
            reconstruction_loss = jnp.mean(jnp.square(recon_micro - h_micro))
            
            # 7. Topological Sparsity
            topo_sparsity_loss = sum(jnp.mean(jnp.abs(adj)) for adj in all_coarse_adjs)

            # 8. Symbolic Sparsity (L1 on coefficients)
            symbolic_sparsity_loss = jnp.mean(jnp.abs(global_coeffs))

            # Dynamic Multi-Task Weighting
            def weighted_loss(L, logv):
                return jnp.exp(-logv) * L + logv
                
            total_loss = (
                weighted_loss(task_loss, logvars['task']) +
                weighted_loss(ce_loss, logvars['ce']) +
                weighted_loss(continuity_loss, logvars['continuity']) +
                weighted_loss(sparsity_loss, logvars['sparsity']) +
                weighted_loss(symbolic_loss, logvars['symbolic']) +
                weighted_loss(reconstruction_loss, logvars['reconstruction']) * 0.1 +
                weighted_loss(topo_sparsity_loss, logvars['topo_sparsity']) +
                weighted_loss(symbolic_sparsity_loss, logvars['sparsity']) 
            )
            
            return total_loss

        @jax.jit
        def _step(st, g, t, r, mei):
            grads = jax.grad(loss_fn)(st.params, g, t, r, mei)
            return st.apply_gradients(grads=grads)
            
        t_input = targets if targets is not None else jnp.zeros((0, 1))
        return _step(state, batch_graphs, t_input, rng, self.micro_ei_baseline)