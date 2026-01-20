import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph
from typing import Callable, Iterable, Optional, Tuple

class GNNLayer(nn.Module):
    """A simple GNN layer using jraph."""
    update_node_fn: Callable
    update_edge_fn: Optional[Callable] = None
    aggregate_edges_for_nodes_fn: Callable = jraph.segment_sum

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        gn = jraph.GraphNetwork(
            update_node_fn=self.update_node_fn,
            update_edge_fn=self.update_edge_fn,
            aggregate_edges_for_nodes_fn=self.aggregate_edges_for_nodes_fn
        )
        return gn(graph)

class MLP(nn.Module):
    features: Iterable[int]
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i < len(self.features) - 1 or self.activate_final:
                x = nn.relu(x)
        return x

class IterativeDecimator(nn.Module):
    """
    Learnable pooling layer for renormalization (Rs operator).
    Performs soft-clustering of nodes to coarse-grain the graph.
    """
    latent_dim: int
    num_clusters: int

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        # graph.nodes shape: [num_nodes, node_feat]
        node_feats = graph.nodes
        
        # Predict cluster assignment probabilities (Soft assignment)
        # We use an MLP to compute assignment logits
        assignment_logits = MLP([32, self.num_clusters])(node_feats)
        assignments = nn.softmax(assignment_logits, axis=-1) # [num_nodes, num_clusters]
        
        # Coarse-grain node features: M = S^T * H
        coarse_nodes = jnp.matmul(assignments.T, node_feats) # [num_clusters, node_feat]
        
        # Coarse-grain adjacency matrix (edges)
        # In a real GNN, we'd also transform the edges. For now, we assume fully connected
        # or simplified graph structures for the demonstration.
        
        # Return coarse graph and assignment matrix for loss calculation
        coarse_graph = graph._replace(
            nodes=coarse_nodes,
            n_node=jnp.array([self.num_clusters]),
            # Reset edges for simplicity in this baseline
            senders=jnp.array([]),
            receivers=jnp.array([]),
            edges=None,
            n_edge=jnp.array([0])
        )
        
        return coarse_graph, assignments

class MSVIB(nn.Module):
    """Multi-Scale Variational Information Bottleneck."""
    encoder_features: Iterable[int]
    latent_dim: int
    num_clusters: int
    output_dim: int = 1

    def setup(self):
        self.encoder = GNNLayer(
            update_node_fn=lambda n, s, r, g: MLP(self.encoder_features)(n)
        )
        self.decimator = IterativeDecimator(
            latent_dim=self.latent_dim,
            num_clusters=self.num_clusters
        )
        # Latent distributions for VIB
        self.fc_mu = nn.Dense(self.latent_dim)
        self.fc_logvar = nn.Dense(self.latent_dim)
        
        # Predictor head
        self.predictor = MLP([32, self.output_dim])

    def __call__(self, graph: jraph.GraphsTuple):
        # 1. Encode Micro-state
        h = self.encoder(graph)
        
        # 2. Renormalize (Coarse-grain)
        coarse_graph, assignments = self.decimator(h)
        
        # 3. Variational Bottleneck on Macro-state
        # Global pooling of coarse nodes
        macro_summary = jnp.mean(coarse_graph.nodes, axis=0)
        
        mu = self.fc_mu(macro_summary)
        logvar = self.fc_logvar(macro_summary)
        
        # Reparameterization trick
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(self.make_rng('vmap_rng') if self.has_rng('vmap_rng') else jax.random.PRNGKey(0), mu.shape)
        z = mu + eps * std
        
        # 4. Predict outcome (e.g. future state or property)
        pred_y = self.predictor(z)
        
        return mu, logvar, pred_y, assignments, coarse_graph
