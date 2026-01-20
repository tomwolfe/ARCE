import jax
import jax.numpy as jnp

def kl_divergence(mu, logvar):
    """Standard KL divergence for VIB."""
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1)

def effective_information(transition_matrix):
    """
    Calculates Effective Information (EI) of a transition matrix.
    EI = H(average(W_i)) - average(H(W_i))
    where W_i are the rows of the transition matrix.
    """
    # Ensure transition matrix is normalized
    W = transition_matrix / (jnp.sum(transition_matrix, axis=-1, keepdims=True) + 1e-8)
    
    # H(average(W_i))
    avg_W = jnp.mean(W, axis=0)
    h_avg_W = -jnp.sum(avg_W * jnp.log2(avg_W + 1e-8))
    
    # average(H(W_i))
    h_W_i = -jnp.sum(W * jnp.log2(W + 1e-8), axis=-1)
    avg_h_W = jnp.mean(h_W_i)
    
    return h_avg_W - avg_h_W

def soft_histogram2d(x, y, bins=10, range_lims=[[-3.0, 3.0], [-3.0, 3.0]], bandwidth=0.1):
    """
    Differentiable 2D histogram using Gaussian kernels.
    """
    x_min, x_max = range_lims[0]
    y_min, y_max = range_lims[1]
    
    x_centers = jnp.linspace(x_min, x_max, bins)
    y_centers = jnp.linspace(y_min, y_max, bins)
    
    # Compute distances to centers: [N, bins]
    x_dist = jnp.square(x[:, None] - x_centers[None, :])
    y_dist = jnp.square(y[:, None] - y_centers[None, :])
    
    # Gaussian kernel: [N, bins]
    x_weights = jnp.exp(-x_dist / (2 * bandwidth**2))
    x_weights /= (jnp.sum(x_weights, axis=-1, keepdims=True) + 1e-8)
    
    y_weights = jnp.exp(-y_dist / (2 * bandwidth**2))
    y_weights /= (jnp.sum(y_weights, axis=-1, keepdims=True) + 1e-8)
    
    # Sum of outer products: [bins, bins]
    return jnp.matmul(x_weights.T, y_weights)

def estimate_transition_matrix(latent_series, num_bins=10):
    """
    Estimates a Markov transition matrix from a sequence of latent states.
    Uses differentiable soft-binning for JAX gradient compatibility.
    """
    # Use only the first dimension for state estimation as in the original
    data = latent_series[:, 0]
    
    # Create transitions (t -> t+1)
    x = data[:-1]
    y = data[1:]
    
    # Use fixed range for JIT compatibility
    range_lims = [[-3.0, 3.0], [-3.0, 3.0]]
    matrix = soft_histogram2d(x, y, bins=num_bins, range_lims=range_lims)
    
    # Normalize rows
    row_sums = jnp.sum(matrix, axis=-1, keepdims=True)
    matrix = matrix / (row_sums + 1e-8)
    
    return matrix

def causal_emergence_loss(micro_ei, macro_ei):
    """
    Loss to maximize Causal Emergence.
    We want macro_ei > micro_ei.
    """
    return jax.nn.relu(micro_ei - macro_ei)

def information_bottleneck_loss(mu, logvar, pred_y, target_y, beta=1e-3):
    """
    VIB Loss: Reconstruction Loss + beta * KL Divergence
    """
    recon_loss = jnp.mean(jnp.square(pred_y - target_y))
    kl_loss = jnp.mean(kl_divergence(mu, logvar))
    return recon_loss + beta * kl_loss
