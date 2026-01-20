import jax
import jax.numpy as jnp
import jraph
from arce.engine import ARCE
from arce.utils import ising_to_jraph

def test_reconstruction_flow():
    config = {
        'latent_dim': 8,
        'num_clusters_list': [8, 4],
        'encoder_features': [32, 32],
        'lr': 1e-3
    }
    engine = ARCE(config)
    
    # Create a dummy graph
    L = 6
    spins = jax.random.choice(jax.random.PRNGKey(42), jnp.array([-1.0, 1.0]), shape=(L, L))
    graph = ising_to_jraph(spins)
    
    rng = jax.random.PRNGKey(0)
    engine.init_model(rng, graph)
    
    # Test that we can call coarsen and get recon_micro
    # We need to manually call the model to check recon_micro since coarsen doesn't return it yet
    # (Actually, I updated coarsen in engine.py to return coarse_graph, mu only. 
    # Let's check engine.py again)
    
    model_params = {k: v for k, v in engine.state.params.items() if k != 'loss_logvars'}
    mu, logvar, pred_y, assignments, coarse_graph, recon_micro, h_micro = engine.model.apply(
        {'params': model_params}, 
        graph,
        rngs={'vmap_rng': rng}
    )
    
    assert recon_micro.shape == h_micro.shape
    assert not jnp.allclose(recon_micro, h_micro) # Initially should be different
    
    # Test that gradients flow from reconstruction loss to params
    def recon_loss_fn(params):
        m_params = {k: v for k, v in params.items() if k != 'loss_logvars'}
        _, _, _, _, _, r_micro, h_m = engine.model.apply(
            {'params': m_params}, 
            graph,
            rngs={'vmap_rng': rng}
        )
        return jnp.mean(jnp.square(r_micro - h_m))
    
    grads = jax.grad(recon_loss_fn)(engine.state.params)
    # Check that some gradients are non-zero for the decoder
    # The decoder should be at the top level of params
    assert 'decoder' in grads
    # Recursively check if there are any non-zero gradients in the decoder subtree
    def has_nonzero(tree):
        return jnp.any(jax.tree_util.tree_reduce(lambda a, b: a | (jnp.any(b != 0)), tree, False))
    
    assert has_nonzero(grads['decoder'])

if __name__ == "__main__":
    test_reconstruction_flow()
