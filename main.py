import jax
import jax.numpy as jnp
from arce.engine import ARCE
from arce.utils import generate_ising_2d, ising_to_jraph, plot_information_tradeoff
import numpy as np

def main():
    print("--- ARCE: Automated Renormalization & Coarse-Graining Engine ---")
    
    # 1. Configuration
    config = {
        'encoder_features': [32, 16],
        'latent_dim': 4,
        'num_clusters': 4,
        'lr': 1e-3
    }
    
    # 2. Initialize Engine
    engine = ARCE(config)
    rng = jax.random.PRNGKey(42)
    
    # Generate a sample graph to initialize params
    sample_spins = generate_ising_2d(L=4, T=2.26)
    sample_graph = ising_to_jraph(sample_spins)
    engine.init_model(rng, sample_graph)
    
    print("Engine Initialized.")
    
    # 3. Demonstration: Coarsening
    # Generate data at different temperatures
    temps = [1.0, 2.26, 4.0] # Low, Critical, High
    for T in temps:
        spins = generate_ising_2d(L=10, T=T)
        graph = ising_to_jraph(spins)
        
        coarse_graph, mu = engine.coarsen(graph)
        
        magnetization = np.abs(np.mean(spins))
        print(f"Temp: {T:.2f} | Micro-Magnetization: {magnetization:.4f} | Macro-Latent Mean: {mu.mean():.4f}")

    # 4. Symbolic Discovery (Skeleton)
    # Simulate a small time series
    history = [ising_to_jraph(generate_ising_2d(L=10, T=2.26)) for _ in range(5)]
    equation = engine.discover_dynamics(history)
    print(f"Discovered Equation: {equation}")

    # 5. Probabilistic Manifold Mapping
    _, last_mu = engine.coarsen(history[-1])
    prediction = engine.predict_manifold(last_mu, horizon=5)
    print(f"Predicted Final State Mean: {prediction['mean'].mean():.4f}")

    # 6. Information Trade-off Visualization (Mock data for demo)
    betas = [1e-4, 1e-3, 1e-2, 0.1, 0.5]
    info_loss = [0.1, 0.5, 1.2, 2.5, 4.0]
    predictive_power = [0.95, 0.92, 0.85, 0.70, 0.50]
    plot_information_tradeoff(betas, info_loss, predictive_power)
    print("Visualization saved to info_tradeoff.png")

if __name__ == "__main__":
    main()
