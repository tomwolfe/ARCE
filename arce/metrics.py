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

def soft_histogram2d(x, y, bins=10, bandwidth=None, adaptive_range=True):
    """
    Differentiable 2D histogram with optional adaptive range and bandwidth.
    If bandwidth is None, it uses a heuristic based on the data scale.
    """
    if adaptive_range:
        x_min, x_max = jnp.min(x), jnp.max(x)
        y_min, y_max = jnp.min(y), jnp.max(y)
        # Add some padding to avoid boundary effects
        x_range = jnp.maximum(x_max - x_min, 1e-5)
        y_range = jnp.maximum(y_max - y_min, 1e-5)
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
    else:
        x_min, x_max = -3.0, 3.0
        y_min, y_max = -3.0, 3.0
    
    x_centers = jnp.linspace(x_min, x_max, bins)
    y_centers = jnp.linspace(y_min, y_max, bins)
    
    # Adaptive bandwidth if not provided
    if bandwidth is None:
        # Silverman's rule of thumb inspired adaptive bandwidth
        n_samples = x.shape[0]
        # Use robust variance to avoid NaN gradients at zero variance
        var_x = jnp.var(x)
        var_y = jnp.var(y)
        std_x = jnp.sqrt(var_x + 1e-9)
        std_y = jnp.sqrt(var_y + 1e-9)
        
        # Rule of thumb with a floor to prevent "emptying out"
        bandwidth_x = 1.06 * std_x * (n_samples**(-1/5))
        bandwidth_y = 1.06 * std_y * (n_samples**(-1/5))
        
        # Normalize by range for the Gaussian kernel calculation
        # and ensure a minimum relative bandwidth (e.g., 5% of range)
        range_x = x_max - x_min + 1e-5
        range_y = y_max - y_min + 1e-5
        bandwidth_x = jnp.maximum(bandwidth_x / range_x, 0.05)
        bandwidth_y = jnp.maximum(bandwidth_y / range_y, 0.05)
    else:
        bandwidth_x = bandwidth
        bandwidth_y = bandwidth

    # Compute distances to centers: [N, bins]
    # Scaled by range to prevent vanishing gradients if range is large
    x_dist = jnp.square((x[:, None] - x_centers[None, :]) / (x_max - x_min + 1e-5))
    y_dist = jnp.square((y[:, None] - y_centers[None, :]) / (y_max - y_min + 1e-5))
    
    # Gaussian kernel with adaptive or fixed bandwidth
    # Added small constant to weights to ensure non-zero gradients everywhere
    eps_grad = 1e-4
    x_weights = jnp.exp(-x_dist / (2 * jnp.maximum(bandwidth_x, 1e-3)**2)) + eps_grad
    x_weights /= (jnp.sum(x_weights, axis=-1, keepdims=True) + 1e-8)
    
    y_weights = jnp.exp(-y_dist / (2 * jnp.maximum(bandwidth_y, 1e-3)**2)) + eps_grad
    y_weights /= (jnp.sum(y_weights, axis=-1, keepdims=True) + 1e-8)
    
    return jnp.matmul(x_weights.T, y_weights)

def estimate_transition_matrix(latent_series, num_bins=12):
    """
    Estimates a Markov transition matrix. Improved with kernel density smoothing
    and adaptive binning to reduce sensitivity to 'num_bins'.
    Now uses all latent dimensions to avoid heuristic bias.
    """
    dim = latent_series.shape[1]
    
    matrices = []
    # Use all dimensions (Point 3)
    for i in range(dim):
        data = latent_series[:, i]
        x, y = data[:-1], data[1:]
        m = soft_histogram2d(x, y, bins=num_bins, bandwidth=None, adaptive_range=True)
        # Add small epsilon for Laplace smoothing (robustness to sparse data)
        m = m + 1e-6
        # Normalize rows
        m = m / jnp.sum(m, axis=-1, keepdims=True)
        matrices.append(m)
    
    return jnp.stack(matrices).mean(axis=0)


def causal_emergence_loss(micro_ei, macro_ei, mi_micro_macro, target_mi=1.0):
    """
    Loss to maximize Causal Emergence.
    Includes a Mutual Information (MI) constraint between micro and macro levels
    to prevent "hallucinated emergence" where the model ignores the micro-state variance.
    """
    # Maximize (macro_ei - micro_ei) -> Minimize (micro_ei - macro_ei)
    ce = jax.nn.relu(micro_ei - macro_ei)
    
    # MI constraint: Ensure the macro-state captures sufficient information from micro-state
    # This prevents the model from collapsing into a trivial deterministic state.
    mi_penalty = jax.nn.relu(target_mi - mi_micro_macro)
    
    # Grounding: ensure macro_ei doesn't drop to zero
    grounding_penalty = jax.nn.relu(0.5 - macro_ei) 
    
    return ce + 0.5 * mi_penalty + 0.1 * grounding_penalty

def information_bottleneck_loss(mu, logvar, pred_y, target_y, beta=1e-3):
    """
    VIB Loss: Reconstruction Loss + beta * KL Divergence
    """
    recon_loss = jnp.mean(jnp.square(pred_y - target_y))
    kl_loss = jnp.mean(kl_divergence(mu, logvar))
    return recon_loss + beta * kl_loss
