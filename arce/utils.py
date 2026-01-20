import numpy as np
import jax.numpy as jnp
import jraph
import networkx as nx

def generate_ising_2d(L=10, T=2.26, steps=1000):
    """
    Generates 2D Ising model configurations using Metropolis algorithm.
    T_c approx 2.269
    """
    spins = np.random.choice([-1, 1], size=(L, L))
    
    for _ in range(steps):
        i, j = np.random.randint(0, L, size=2)
        # Periodic boundary conditions
        delta_E = 2 * spins[i, j] * (
            spins[(i+1)%L, j] + spins[(i-1)%L, j] +
            spins[i, (j+1)%L] + spins[i, (j-1)%L]
        )
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T):
            spins[i, j] *= -1
            
    return spins

def ising_to_jraph(spins):
    """Converts a 2D spin lattice to a jraph GraphsTuple."""
    L = spins.shape[0]
    num_nodes = L * L
    
    # Nodes are spins (flattened)
    nodes = spins.flatten().reshape(-1, 1).astype(np.float32)
    
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
