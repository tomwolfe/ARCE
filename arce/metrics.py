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

def estimate_transition_matrix(latent_series, num_bins=10):
    """
    Estimates a Markov transition matrix from a sequence of latent states.
    Uses simple histogram binning for demonstration.
    """
    # Flatten to 1D for simple binning if necessary, 
    # or handle multidimensional binning.
    # Here we take the first dimension as the 'state'.
    data = latent_series[:, 0]
    
    # Map data to bins
    min_val, max_val = jnp.min(data), jnp.max(data)
    bins = jnp.linspace(min_val, max_val, num_bins + 1)
    
    indices = jnp.digitize(data, bins) - 1
    indices = jnp.clip(indices, 0, num_bins - 1)
    
    # Count transitions
    matrix = jnp.zeros((num_bins, num_bins))
    
    def update_matrix(m, idx_pair):
        i, j = idx_pair
        return m.at[i, j].add(1)
    
    # This is a bit slow in pure JAX without scan/vmap, 
    # but for demo it's fine.
    for t in range(len(indices) - 1):
        matrix = update_matrix(matrix, (indices[t], indices[t+1]))
        
    # Normalize rows
    row_sums = jnp.sum(matrix, axis=-1, keepdims=True)
    matrix = matrix / (row_sums + 1e-8)
    
    return matrix

def causal_emergence_loss(micro_ei, macro_ei):
    """
    Loss to maximize Causal Emergence.
    We want macro_ei > micro_ei.
    """
    return jnp.relu(micro_ei - macro_ei)

def information_bottleneck_loss(mu, logvar, pred_y, target_y, beta=1e-3):
    """
    VIB Loss: Reconstruction Loss + beta * KL Divergence
    """
    recon_loss = jnp.mean(jnp.square(pred_y - target_y))
    kl_loss = jnp.mean(kl_divergence(mu, logvar))
    return recon_loss + beta * kl_loss
