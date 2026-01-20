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

    def init_model(self, rng, sample_graph):
        variables = self.model.init(rng, sample_graph)
        params = variables['params']
        
        # Add learnable loss weights (log variances for uncertainty weighting)
        params['loss_logvars'] = {
            'task': jnp.zeros(()),
            'ce': jnp.zeros(()),
            'continuity': jnp.zeros(()),
            'sparsity': jnp.zeros(())
        }
        
        self.params = params
        tx = optax.adam(learning_rate=self.config.get('lr', 1e-3))
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=tx
        )

    def coarsen(self, micro_data: jraph.GraphsTuple, rng=None):
        """Returns optimal macro-variables."""
        rngs = {'vmap_rng': rng} if rng is not None else {}
        # Filter out loss_logvars from params for model call
        model_params = {k: v for k, v in self.state.params.items() if k != 'loss_logvars'}
        mu, logvar, pred_y, assignments, coarse_graph = self.model.apply(
            {'params': model_params}, 
            micro_data,
            rngs=rngs
        )
        return coarse_graph, mu

    def get_basis_features(self, data, include_transcendental=True):
        """Configurable basis library for Symbolic Discovery."""
        n, d = data.shape
        # Linear and Bias
        feats = [jnp.ones((n, 1)), data]
        names = ["1"] + [f"M{i}" for i in range(d)]
        
        # Quadratic
        for i in range(d):
            for j in range(i, d):
                feats.append((data[:, i] * data[:, j])[:, None])
                names.append(f"M{i}*M{j}")
        
        if include_transcendental:
            # Transcendental (expanded basis)
            feats.append(jnp.sin(data))
            names += [f"sin(M{i})" for i in range(d)]
            feats.append(jnp.cos(data))
            names += [f"cos(M{i})" for i in range(d)]
            feats.append(jnp.exp(-jnp.square(data))) # Gaussian kernels
            names += [f"exp(-M{i}^2)" for i in range(d)]
            
        return jnp.concatenate(feats, axis=-1), names

    def discover_dynamics(self, time_series_graphs):
        """
        Identifies sparse dynamics in the coarse-grained latent space.
        Uses JAX-native ISTA (Lasso) for end-to-end differentiability and speed.
        """
        print("ARCE: Identifying Sufficient Statistics (JAX-native)...")
        
        # 1. Extract macro-states over time
        macro_states = []
        for g in time_series_graphs:
            _, mu = self.coarsen(g)
            macro_states.append(mu)
        
        macro_states = jnp.concatenate(macro_states, axis=0) # [time, latent_dim]
        
        # 2. Prepare data for Symbolic Regression (X, y)
        X = macro_states[:-1]
        y = macro_states[1:] - macro_states[:-1]
        
        # 3. Features from configurable library
        X_poly, names = self.get_basis_features(X)

        # 4. JAX-native Sparse Regression (ISTA)
        def soft_threshold(x, thresh):
            return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0)

        def lasso_ista(X_mat, y_mat, alpha=0.01, num_iters=1000):
            n_feat = X_mat.shape[1]
            n_target = y_mat.shape[1]
            # Use a slightly more robust step size calculation
            step = 0.5 / (jnp.linalg.norm(X_mat, ord=2)**2 + 1e-5)
            
            def body(i, w):
                grad = X_mat.T @ (X_mat @ w - y_mat)
                return soft_threshold(w - step * grad, alpha * step)
                
            return jax.lax.fori_loop(0, num_iters, body, jnp.zeros((n_feat, n_target)))

        coeffs = lasso_ista(X_poly, y, alpha=self.config.get('ista_alpha', 0.01))
        
        # 5. Format the discovered equation
        equations = []
        for i in range(y.shape[1]):
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
            model_params = {k: v for k, v in params.items() if k != 'loss_logvars'}
            
            mu, logvar, pred_y, _, _ = self.model.apply(
                {'params': model_params}, 
                graphs,
                rngs={'vmap_rng': r}
            )
            
            # 1. Causal Emergence Loss (Robust)
            macro_ei = effective_information_robust(
                mu, 
                use_gaussian=self.config.get('use_gaussian_ei', False)
            )
            ce_loss = causal_emergence_loss(micro_ei, macro_ei)
            
            # Regularization to prevent "hallucinated" causal links:
            latent_reg = 0.01 * jnp.mean(jnp.square(mu)) + 0.01 * jnp.mean(kl_divergence(mu, logvar))
            ce_loss = ce_loss + latent_reg
            
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
            
            # Dynamic Multi-Task Weighting
            def weighted_loss(L, logv):
                return jnp.exp(-logv) * L + logv
                
            total_loss = (
                weighted_loss(task_loss, logvars['task']) +
                weighted_loss(ce_loss, logvars['ce']) +
                weighted_loss(continuity_loss, logvars['continuity']) +
                weighted_loss(sparsity_loss, logvars['sparsity'])
            )
            
            return total_loss

        @jax.jit
        def _step(st, g, t, r, mei):
            grads = jax.grad(loss_fn)(st.params, g, t, r, mei)
            return st.apply_gradients(grads=grads)
            
        t_input = targets if targets is not None else jnp.zeros((0, 1))
        return _step(state, batch_graphs, t_input, rng, self.micro_ei_baseline)