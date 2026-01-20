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

def effective_information_gaussian(latent_series):
    """
    Calculates Effective Information using a Gaussian approximation for joint distributions.
    EI = I(Z_t; Z_{t+1}) = H(Z_{t+1}) - H(Z_{t+1} | Z_t)
    This avoids the independence assumption and is O(D^3) in latent dimension D.
    """
    z_t = latent_series[:-1]
    z_next = latent_series[1:]
    
    # Concatenate to get joint distribution
    joint = jnp.concatenate([z_t, z_next], axis=-1)
    
    # Covariance matrices
    cov_joint = jnp.cov(joint, rowvar=False)
    d = z_t.shape[1]
    
    cov_t = cov_joint[:d, :d]
    cov_next = cov_joint[d:, d:]
    
    # Add small diagonal for stability
    eps = 1e-6 * jnp.eye(cov_joint.shape[0])
    
    # MI = 0.5 * log(|Cov_t| * |Cov_next| / |Cov_joint|)
    # Use slogdet for stability
    sign_t, logdet_t = jnp.linalg.slogdet(cov_t + eps[:d, :d])
    sign_next, logdet_next = jnp.linalg.slogdet(cov_next + eps[d:, d:])
    sign_joint, logdet_joint = jnp.linalg.slogdet(cov_joint + eps)
    
    mi = 0.5 * (logdet_t + logdet_next - logdet_joint)
    return jnp.maximum(mi, 0.0)

def effective_information_robust(latent_series, num_bins=16, use_gaussian=False):
    """
    Robust EI estimator that selects between Gaussian and Non-parametric (binning).
    Non-parametric version is better for phase transitions (e.g. Ising at Tc).
    """
    if use_gaussian:
        return effective_information_gaussian(latent_series)
    
    # Non-parametric EI via transition matrix estimation
    trans_matrix = estimate_transition_matrix(latent_series, num_bins=num_bins)
    return effective_information(trans_matrix)

def soft_histogram2d(x, y, bins=10, bandwidth=0.1, adaptive_range=True):
    """
    Differentiable 2D histogram with optional adaptive range.
    """
    if adaptive_range:
        x_min, x_max = jnp.min(x), jnp.max(x)
        y_min, y_max = jnp.min(y), jnp.max(y)
        # Add some padding
        x_min -= 0.1 * (x_max - x_min + 1e-5)
        x_max += 0.1 * (x_max - x_min + 1e-5)
        y_min -= 0.1 * (y_max - y_min + 1e-5)
        y_max += 0.1 * (y_max - y_min + 1e-5)
    else:
        x_min, x_max = -3.0, 3.0
        y_min, y_max = -3.0, 3.0
    
    x_centers = jnp.linspace(x_min, x_max, bins)
    y_centers = jnp.linspace(y_min, y_max, bins)
    
    # Compute distances to centers: [N, bins]
    # Scaled by range to prevent vanishing gradients if range is large
    x_dist = jnp.square((x[:, None] - x_centers[None, :]) / (x_max - x_min + 1e-5))
    y_dist = jnp.square((y[:, None] - y_centers[None, :]) / (y_max - y_min + 1e-5))
    
    # Gaussian kernel
    x_weights = jnp.exp(-x_dist / (2 * bandwidth**2))
    x_weights /= (jnp.sum(x_weights, axis=-1, keepdims=True) + 1e-8)
    
    y_weights = jnp.exp(-y_dist / (2 * bandwidth**2))
    y_weights /= (jnp.sum(y_weights, axis=-1, keepdims=True) + 1e-8)
    
    return jnp.matmul(x_weights.T, y_weights)

def estimate_transition_matrix(latent_series, num_bins=12):
    """
    Estimates a Markov transition matrix. Improved with kernel density smoothing
    and adaptive binning to reduce sensitivity to 'num_bins'.
    """
    dim = latent_series.shape[1]
    
    # We use a more robust bandwidth selection (Silverman's rule-of-thumb inspired)
    n_samples = latent_series.shape[0]
    bandwidth = 1.06 * jnp.std(latent_series) * (n_samples**(-1/5))
    bandwidth = jnp.maximum(bandwidth, 0.05) # Floor
    
    matrices = []
    # Average across dimensions to get a robust scalar transition estimate
    for i in range(min(dim, 4)):
        data = latent_series[:, i]
        x, y = data[:-1], data[1:]
        m = soft_histogram2d(x, y, bins=num_bins, bandwidth=bandwidth, adaptive_range=True)
        # Add small epsilon for Laplace smoothing (robustness to sparse data)
        m = m + 1e-6
        # Normalize rows
        m = m / jnp.sum(m, axis=-1, keepdims=True)
        matrices.append(m)
    
    return jnp.stack(matrices).mean(axis=0)


def causal_emergence_loss(micro_ei, macro_ei, macro_entropy_weight=0.1):
    """
    Loss to maximize Causal Emergence.
    Includes a 'Grounding' term: we want macro_ei > micro_ei, but we also
    want the macro-state to have sufficient entropy (avoiding collapse to trivial states).
    """
    ce = jax.nn.relu(micro_ei - macro_ei)
    
    # Penalty for too low macro-EI or collapsed entropy
    # This addresses the 'Causal Emergence Paradox' by preventing trivial solutions.
    grounding_penalty = jax.nn.relu(0.5 - macro_ei) 
    
    return ce + 0.1 * grounding_penalty

def information_bottleneck_loss(mu, logvar, pred_y, target_y, beta=1e-3):
    """
    VIB Loss: Reconstruction Loss + beta * KL Divergence
    """
    recon_loss = jnp.mean(jnp.square(pred_y - target_y))
    kl_loss = jnp.mean(kl_divergence(mu, logvar))
    return recon_loss + beta * kl_loss
