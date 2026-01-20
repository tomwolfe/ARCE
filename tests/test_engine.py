import jax
import jax.numpy as jnp
import jraph
from arce.engine import ARCE
from arce.utils import ising_to_jraph

def test_arce_init():
    config = {
        'latent_dim': 8,
        'num_clusters_list': [4],
        'lr': 1e-3
    }
    engine = ARCE(config)
    
    # Create a small dummy graph
    spins = jnp.ones((4, 4))
    graph = ising_to_jraph(spins)
    
    rng = jax.random.PRNGKey(0)
    engine.init_model(rng, graph)
    
    assert engine.state is not None
    assert 'loss_logvars' in engine.state.params

def test_train_step():
    config = {
        'latent_dim': 4,
        'num_clusters_list': [2],
        'lr': 1e-3,
        'ista_alpha': 0.01
    }
    engine = ARCE(config)
    
    spins = jnp.ones((4, 4))
    graph = ising_to_jraph(spins)
    # Batch of 2 graphs
    batch_graph = jraph.batch([graph, graph])
    
    rng = jax.random.PRNGKey(0)
    engine.init_model(rng, batch_graph)
    
    targets = jnp.ones((2, 1))
    
    initial_params = engine.state.params
    new_state = engine.train_step(engine.state, batch_graph, targets, rng)
    
    # Verify params updated
    assert not jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), initial_params, new_state.params)
    )

def test_spectrum_analysis():
    config = {}
    engine = ARCE(config)
    
    L = 4
    spins = jnp.ones((L, L))
    graph = ising_to_jraph(spins)
    
    n_suggested = engine.analyze_graph_spectrum(graph)
    assert isinstance(n_suggested, int)
    assert n_suggested > 0

def test_differentiable_ista():
    # Test that gradients flow through _lasso_ista_jax
    X = jax.random.normal(jax.random.PRNGKey(0), (10, 5))
    y = jax.random.normal(jax.random.PRNGKey(1), (10, 2))
    
    def loss(X_val):
        coeffs = ARCE._lasso_ista_jax(X_val, y, alpha=0.1, num_iters=10)
        return jnp.sum(jnp.square(coeffs))
    
    grads = jax.grad(loss)(X)
    assert jnp.any(grads != 0)

def test_soft_histogram_diff():
    from arce.metrics import soft_histogram2d
    x = jax.random.normal(jax.random.PRNGKey(0), (100,))
    y = jax.random.normal(jax.random.PRNGKey(1), (100,))
    
    def loss(x_val):
        h = soft_histogram2d(x_val, y, bins=10)
        return jnp.sum(jnp.square(h))
    
    grads = jax.grad(loss)(x)
    assert jnp.any(grads != 0)
