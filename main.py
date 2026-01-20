import jax
import jax.numpy as jnp
from arce.engine import ARCE
from arce.utils import generate_ising_2d, ising_to_jraph, plot_information_tradeoff
import numpy as np
import jraph

def train_arce(engine, num_epochs=20):
    print("ARCE: Training on Ising Data...")
    rng = jax.random.PRNGKey(42)
    
    # Generate training data: configurations at various temperatures
    train_sequences = []
    targets = []
    
    for T in [1.5, 2.26, 3.5]:
        for _ in range(3): # 3 sequences per temperature
            seq = []
            rng, subkey = jax.random.split(rng)
            spins = generate_ising_2d(L=8, T=T, steps=200, key=subkey)
            for _ in range(10): # Sequence length 10
                graph = ising_to_jraph(spins)
                seq.append(graph)
                
                # Evolve system
                rng, subkey = jax.random.split(rng)
                spins = generate_ising_2d(L=8, T=T, steps=20, key=subkey)
                
            train_sequences.append(seq)
            targets.append(jnp.abs(jnp.mean(spins)))
            
    targets = jnp.array(targets).reshape(-1, 1)
    
    # Metrics for plotting
    mi_list = []
    task_loss_list = []

    # Training Loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(len(train_sequences)):
            # Convert sequence to a single padded GraphsTuple for processing
            batch_graph = jraph.batch(train_sequences[i])
            rng, step_rng = jax.random.split(rng)
            # UNSUPERVISED: We pass None for targets to encourage discovery
            engine.state, aux = engine.train_step(
                engine.state, 
                batch_graph, 
                None, 
                step_rng
            )
            
            if epoch == num_epochs - 1:
                mi_list.append(float(aux['mi']))
                task_loss_list.append(float(aux['task']))

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {float(aux['total']):.4f} | CE: {float(aux['ce']):.4f} | Recon: {float(aux['recon']):.4f}")

    return mi_list, task_loss_list

def main():
    print("--- ARCE: Automated Renormalization & Coarse-Graining Engine ---")
    
    # 1. Configuration
    config = {
        'encoder_features': [32, 16], # Slightly larger networks
        'latent_dim': 2,
        'num_clusters_list': [8, 4], # Hierarchical pooling: 64 -> 8 -> 4 nodes
        'lr': 1e-3,
        'top_k_edges': 3,            # Sparse coarsening: keep top 3 edges per cluster
        'use_gaussian_ei': True,      # Use Gaussian EI for more stable training gradients
        'ista_alpha': 0.05            # Sparsity penalty for symbolic discovery
    }
    
    # 2. Initialize Engine
    engine = ARCE(config)
    rng = jax.random.PRNGKey(42)
    
    # Generate a sample graph to initialize params
    rng, subkey = jax.random.split(rng)
    sample_spins = generate_ising_2d(L=6, T=2.26, key=subkey)
    sample_graph = ising_to_jraph(sample_spins)
    engine.init_model(rng, sample_graph)
    
    print("Engine Initialized.")
    
    # 3. Functional Training
    mi_data, task_data = train_arce(engine)
    
    # 4. Demonstration: Coarsening
    print("\n--- Demonstration: Coarsening ---")
    temps = [1.0, 2.26, 4.0] 
    for T in temps:
        rng, subkey = jax.random.split(rng)
        spins = generate_ising_2d(L=6, T=T, key=subkey)
        graph = ising_to_jraph(spins)
        
        rng, subkey = jax.random.split(rng)
        coarse_graph, mu = engine.coarsen(graph, rng=subkey)
        
        magnetization = jnp.abs(jnp.mean(spins))
        print(f"Temp: {T:.2f} | Micro-Magnetization: {magnetization:.4f} | Macro-Latent Mean: {mu.mean():.4f}")

    # 5. Symbolic Discovery
    print("\n--- Demonstration: Symbolic Discovery ---")
    history = []
    rng, subkey = jax.random.split(rng)
    curr_spins = generate_ising_2d(L=10, T=2.26, steps=500, key=subkey)
    for _ in range(50):
        history.append(ising_to_jraph(curr_spins))
        rng, subkey = jax.random.split(rng)
        curr_spins = generate_ising_2d(L=10, T=2.26, steps=20, key=subkey)
        
    equation = engine.discover_dynamics(history)
    print(f"Discovered Equation: {equation}")

    # 6. Probabilistic Manifold Mapping
    _, last_mu = engine.coarsen(history[-1])
    prediction = engine.predict_manifold(last_mu, horizon=5)
    print(f"Predicted Final State Mean: {prediction['mean'].mean():.4f}")

    # 7. Information Trade-off Visualization (Real data from training)
    sort_idx = np.argsort(mi_data)
    plot_information_tradeoff(
        np.array(mi_data)[sort_idx], 
        np.array(mi_data)[sort_idx], 
        1.0 - np.array(task_data)[sort_idx]
    )
    print("Visualization saved to info_tradeoff.png")

if __name__ == "__main__":
    main()