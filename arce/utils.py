import jax
import jax.numpy as jnp
import jraph
import networkx as nx

def generate_ising_2d(L=10, T=2.26, steps=1000, key=None):
    """
    Generates 2D Ising model configurations using JAX-native Metropolis algorithm.
    Uses checkerboard updates for parallelism.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    key, subkey = jax.random.split(key)
    spins = jax.random.choice(subkey, jnp.array([-1, 1]), shape=(L, L))
    
    # Precompute indices for checkerboard
    indices = jnp.indices((L, L))
    checkerboard = (indices[0] + indices[1]) % 2
    
    def step_fn(spins, key):
        # One update for each color of the checkerboard
        for color in [0, 1]:
            key, subkey = jax.random.split(key)
            
            # Neighbors (periodic boundary conditions)
            up = jnp.roll(spins, shift=1, axis=0)
            down = jnp.roll(spins, shift=-1, axis=0)
            left = jnp.roll(spins, shift=1, axis=1)
            right = jnp.roll(spins, shift=-1, axis=1)
            
            delta_E = 2 * spins * (up + down + left + right)
            
            # Metropolis criterion
            p_accept = jnp.exp(-delta_E / T)
            accept = jax.random.uniform(subkey, shape=(L, L)) < p_accept
            
            # Only update the current color
            mask = (checkerboard == color)
            spins = jnp.where(mask & ( (delta_E <= 0) | accept ), -spins, spins)
            
        return spins, None

    # Run steps
    keys = jax.random.split(key, steps)
    spins, _ = jax.lax.scan(step_fn, spins, keys)
    
    return spins

def ising_to_jraph(spins):
    """Converts a 2D spin lattice to a jraph GraphsTuple."""
    L = spins.shape[0]
    num_nodes = L * L
    
    # Nodes are spins (flattened)
    nodes = spins.flatten().reshape(-1, 1).astype(jnp.float32)
    
    # Grid edges (4-connectivity)
    senders = []
    receivers = []
    for i in range(L):
        for j in range(L):
            curr = i * L + j
            # Right
            senders.append(curr)
            receivers.append(i * L + (j + 1) % L)
            # Left
            senders.append(curr)
            receivers.append(i * L + (j - 1 + L) % L)
            # Down
            senders.append(curr)
            receivers.append(((i + 1) % L) * L + j)
            # Up
            senders.append(curr)
            receivers.append(((i - 1 + L) % L) * L + j)
            
    return jraph.GraphsTuple(
        nodes=jnp.array(nodes),
        edges=None,
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        n_node=jnp.array([num_nodes]),
        n_edge=jnp.array([len(senders)]),
        globals=None
    )

def laplacian_eigen_spectrum(graph: jraph.GraphsTuple):
    """
    Computes the Eigen-spectrum of the graph Laplacian.
    Useful for determining the optimal number of clusters (macro-nodes).
    """
    num_nodes = graph.n_node[0]
    # Reconstruct adjacency matrix
    adj = jnp.zeros((num_nodes, num_nodes))
    adj = adj.at[graph.senders, graph.receivers].set(1.0)
    
    # Normalized Laplacian: L = I - D^-1/2 * A * D^-1/2
    degree = jnp.sum(adj, axis=1)
    d_inv_sqrt = jnp.where(degree > 0, 1.0 / jnp.sqrt(degree + 1e-8), 0.0)
    d_inv_sqrt_mat = jnp.diag(d_inv_sqrt)
    
    norm_laplacian = jnp.eye(num_nodes) - d_inv_sqrt_mat @ adj @ d_inv_sqrt_mat
    
    # Eigenvalues in ascending order
    eigs = jnp.linalg.eigvalsh(norm_laplacian)
    return eigs

def suggest_num_clusters(eigs, threshold=0.1):
    """
    Suggests the number of clusters based on the 'eigengap' or 
    decay of the spectrum.
    """
    # Look for the gap in eigenvalues
    gaps = jnp.diff(eigs)
    # Heuristic: find where the gap is largest among early eigenvalues
    # or where eigenvalues exceed a threshold.
    n_suggested = jnp.argmax(gaps[:len(eigs)//2]) + 1
    return int(n_suggested)

def plot_information_tradeoff(betas, info_loss, predictive_power):
    """Visualization for Information Loss vs. Predictive Power."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(info_loss, predictive_power, 'o-')
    plt.xlabel("Information Compression (KL Divergence)")
    plt.ylabel("Predictive Power (1 - MSE)")
    plt.title("Information Bottleneck Trade-off Curve")
    plt.grid(True)
    plt.savefig("info_tradeoff.png")
    plt.close()
