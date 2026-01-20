import jax
import jax.numpy as jnp
from arce.engine import ARCE
from arce.utils import generate_ising_2d, ising_to_jraph, plot_information_tradeoff
import numpy as np

def train_arce(engine, num_epochs=20):
    print("ARCE: Training on Ising Data...")
    rng = jax.random.PRNGKey(42)
    
    # Generate training data: configurations at various temperatures
    # Smaller L and fewer samples for speed
    train_data = []
    targets = []
    for T in [1.5, 2.26, 3.5]: # Just 3 representative temps
        for _ in range(3):
            spins = generate_ising_2d(L=6, T=T, steps=200)
            graph = ising_to_jraph(spins)
            train_data.append(graph)
            targets.append(np.abs(np.mean(spins)))
            
    targets = jnp.array(targets).reshape(-1, 1)
    
    # Training Loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(len(train_data)):
            rng, step_rng = jax.random.split(rng)
            engine.state = engine.train_step(
                engine.state, 
                train_data[i], 
                targets[i:i+1], 
                step_rng
            )
            
        if epoch % 5 == 0:
            # Quick loss check on first sample
            mu, logvar, pred_y, _, _ = engine.model.apply(
                {'params': engine.state.params}, 
                train_data[0],
                rngs={'vmap_rng': jax.random.PRNGKey(0)}
            )
            from arce.metrics import information_bottleneck_loss
            loss = information_bottleneck_loss(mu, logvar, pred_y, targets[0:1])
            print(f"Epoch {epoch} | Sample 0 Loss: {loss:.4f}")

def main():
    print("--- ARCE: Automated Renormalization & Coarse-Graining Engine ---")
    
    # 1. Configuration
    config = {
        'encoder_features': [16, 8], # Smaller networks for demo
        'latent_dim': 2,
        'num_clusters': 4,
        'lr': 1e-3
    }
    
    # 2. Initialize Engine
    engine = ARCE(config)
    rng = jax.random.PRNGKey(42)
    
    # Generate a sample graph to initialize params
    sample_spins = generate_ising_2d(L=6, T=2.26)
    sample_graph = ising_to_jraph(sample_spins)
    engine.init_model(rng, sample_graph)
    
    print("Engine Initialized.")
    
    # 3. Functional Training
    train_arce(engine)
    
    # 4. Demonstration: Coarsening
    print("\n--- Demonstration: Coarsening ---")
    temps = [1.0, 2.26, 4.0] 
    for T in temps:
        spins = generate_ising_2d(L=6, T=T)
        graph = ising_to_jraph(spins)
        
        rng, subkey = jax.random.split(rng)
        coarse_graph, mu = engine.coarsen(graph, rng=subkey)
        
        magnetization = np.abs(np.mean(spins))
        print(f"Temp: {T:.2f} | Micro-Magnetization: {magnetization:.4f} | Macro-Latent Mean: {mu.mean():.4f}")

    # 5. Symbolic Discovery
    print("\n--- Demonstration: Symbolic Discovery ---")
    history = [ising_to_jraph(generate_ising_2d(L=10, T=2.26)) for _ in range(10)]
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
