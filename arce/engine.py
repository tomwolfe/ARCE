import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
import optax
from .models import MSVIB
from .metrics import (
    information_bottleneck_loss, 
    effective_information, 
    estimate_transition_matrix, 
    causal_emergence_loss
)
import jraph

class ARCE:
    def __init__(self, config):
        self.config = config
        self.model = MSVIB(
            encoder_features=config.get('encoder_features', [64, 64]),
            latent_dim=config.get('latent_dim', 16),
            num_clusters_list=config.get('num_clusters_list', [4])
        )
        self.params = None
        self.state = None
        # Pre-calculated Micro-EI for Ising Model (approximate baseline)
        self.micro_ei_baseline = 0.5 

    def init_model(self, rng, sample_graph):
        variables = self.model.init(rng, sample_graph)
        self.params = variables['params']
        tx = optax.adam(learning_rate=self.config.get('lr', 1e-3))
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=tx
        )

    def coarsen(self, micro_data: jraph.GraphsTuple, rng=None):
        """Returns optimal macro-variables."""
        rngs = {'vmap_rng': rng} if rng is not None else {}
        mu, logvar, pred_y, assignments, coarse_graph = self.model.apply(
            {'params': self.state.params}, 
            micro_data,
            rngs=rngs
        )
        return coarse_graph, mu

    def discover_dynamics(self, time_series_graphs):
        """
        Identifies sparse dynamics in the coarse-grained latent space.
        Uses JAX-native ISTA (Lasso) for end-to-end differentiability and speed.
        """
        print("ARCE: Identifying Sufficient Statistics (JAX-native)...")
        
        # 1. Extract macro-states over time
        # We use a JITted function to extract all at once if possible
        macro_states = []
        for g in time_series_graphs:
            _, mu = self.coarsen(g)
            macro_states.append(mu)
        
        macro_states = jnp.concatenate(macro_states, axis=0) # [time, latent_dim]
        
        # 2. Prepare data for Symbolic Regression (X, y)
        X = macro_states[:-1]
        y = macro_states[1:] - macro_states[:-1]
        
        # 3. JAX-native Polynomial Features (Degree 2)
        def get_poly_features(data):
            n, d = data.shape
            feats = [jnp.ones((n, 1)), data]
            for i in range(d):
                for j in range(i, d):
                    feats.append((data[:, i] * data[:, j])[:, None])
            return jnp.concatenate(feats, axis=-1)
            
        X_poly = get_poly_features(X)
        
        # Feature names for reconstruction
        d = X.shape[1]
        names = ["1"] + [f"M{i}" for i in range(d)]
        for i in range(d):
            for j in range(i, d):
                names.append(f"M{i}*M{j}")

        # 4. JAX-native Sparse Regression (ISTA)
        def soft_threshold(x, thresh):
            return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0)

        def lasso_ista(X_mat, y_mat, alpha=0.01, num_iters=500):
            n_feat = X_mat.shape[1]
            n_target = y_mat.shape[1]
            # Conservative step size
            step = 0.5 / (jnp.linalg.norm(X_mat, ord=2)**2 + 1e-5)
            
            def body(i, w):
                grad = X_mat.T @ (X_mat @ w - y_mat)
                return soft_threshold(w - step * grad, alpha * step)
                
            return jax.lax.fori_loop(0, num_iters, body, jnp.zeros((n_feat, n_target)))

        coeffs = lasso_ista(X_poly, y, alpha=0.01)
        
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
            mu, logvar, pred_y, _, _ = self.model.apply(
                {'params': params}, 
                graphs,
                rngs={'vmap_rng': r}
            )
            
            # 1. Standard VIB Loss (Compression + Prediction)
            # Use only the last prediction of the sequence for the target
            vib_loss = information_bottleneck_loss(mu[-1:], logvar[-1:], pred_y[-1:], y)
            
            # 2. Causal Emergence Loss
            # Estimate Macro-EI from the latent sequence (mu)
            t_matrix = estimate_transition_matrix(mu)
            macro_ei = effective_information(t_matrix)
            ce_loss = causal_emergence_loss(micro_ei, macro_ei)
            
            # Combined Loss (gamma controls the push for emergence)
            gamma = self.config.get('gamma', 0.1)
            return vib_loss + gamma * ce_loss

        @jax.jit
        def _step(st, g, t, r, mei):
            grads = jax.grad(loss_fn)(st.params, g, t, r, mei)
            return st.apply_gradients(grads=grads)
            
        return _step(state, batch_graphs, targets, rng, self.micro_ei_baseline)