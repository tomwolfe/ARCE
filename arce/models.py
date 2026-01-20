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
    Performs soft-clustering of nodes and contracts edges.
    """
    latent_dim: int
    num_clusters: int

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        # graph.nodes shape: [num_nodes, node_feat]
        node_feats = graph.nodes
        num_nodes = node_feats.shape[0]
        
        # Predict cluster assignment probabilities (Soft assignment)
        assignment_logits = MLP([32, self.num_clusters])(node_feats)
        assignments = nn.softmax(assignment_logits, axis=-1) # [num_nodes, num_clusters]
        
        # 1. Coarse-grain node features: H_macro = S^T * H_micro
        coarse_nodes = jnp.matmul(assignments.T, node_feats) # [num_clusters, node_feat]
        
        # 2. Coarse-grain adjacency matrix (Edge Contraction): A_macro = S^T * A_micro * S
        # Create dense adjacency from senders/receivers
        adj = jnp.zeros((num_nodes, num_nodes))
        if graph.senders is not None and len(graph.senders) > 0:
            adj = adj.at[graph.senders, graph.receivers].set(1.0)
            
        coarse_adj = jnp.matmul(assignments.T, jnp.matmul(adj, assignments))
        
        # Extract new edges from coarse_adj
        # For simplicity in this demo, we keep it as a fully connected graph with weighted edges
        # or we could threshold it. Here we use all non-zero entries.
        c_senders, c_receivers = jnp.nonzero(coarse_adj, size=self.num_clusters**2)
        c_edges = coarse_adj[c_senders, c_receivers][:, jnp.newaxis]
        
        # Return coarse graph and assignment matrix
        coarse_graph = graph._replace(
            nodes=coarse_nodes,
            n_node=jnp.array([self.num_clusters]),
            senders=c_senders,
            receivers=c_receivers,
            edges=c_edges,
            n_edge=jnp.array([len(c_senders)])
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
        # Use segment_mean to handle batches: pooling coarse nodes per graph
        batch_indices = jnp.repeat(
            jnp.arange(len(coarse_graph.n_node)), 
            coarse_graph.n_node, 
            total_repeat_length=coarse_graph.nodes.shape[0]
        )
        macro_summary = jraph.segment_mean(
            coarse_graph.nodes, 
            batch_indices, 
            num_segments=len(coarse_graph.n_node)
        )
        
        mu = self.fc_mu(macro_summary)
        logvar = self.fc_logvar(macro_summary)
        
        # Reparameterization trick
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(
            self.make_rng('vmap_rng') if self.has_rng('vmap_rng') else jax.random.PRNGKey(0), 
            mu.shape
        )
        z = mu + eps * std
        
        # 4. Predict outcome (e.g. future state or property)
        pred_y = self.predictor(z)
        
        return mu, logvar, pred_y, assignments, coarse_graph
