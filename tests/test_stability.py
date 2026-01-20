import jax
import jax.numpy as jnp
from arce.metrics import soft_histogram2d

def test_soft_histogram_stability():
    # Case with zero variance: all samples are identical
    x = jnp.zeros((100,))
    y = jnp.zeros((100,))
    
    # This used to potentially cause issues due to zero bandwidth or vanishing gradients
    h = soft_histogram2d(x, y, bins=10)
    
    assert not jnp.any(jnp.isnan(h))
    assert jnp.all(jnp.isfinite(h))
    assert jnp.abs(jnp.sum(h) - x.shape[0]) < 1e-3 # Should sum to number of samples
    
    # Test gradients in the zero-variance case
    def loss(x_val):
        h_val = soft_histogram2d(x_val, y, bins=10)
        return jnp.sum(jnp.square(h_val))
    
    grads = jax.grad(loss)(x)
    assert not jnp.any(jnp.isnan(grads))
    assert jnp.any(grads != 0) # Should have non-zero gradients thanks to eps_grad

if __name__ == "__main__":
    test_soft_histogram_stability()
