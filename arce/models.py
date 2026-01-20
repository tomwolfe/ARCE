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
    Enhanced to handle batched GraphsTuple.
    """
    num_clusters: int

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        node_feats = graph.nodes
        num_nodes = node_feats.shape[0]
        num_graphs = graph.n_node.shape[0]
        
        # 1. Predict cluster assignment probabilities
        # We want to assign each node to one of 'num_clusters' in its graph.
        assignment_logits = MLP([32, self.num_clusters])(node_feats)
        assignments = nn.softmax(assignment_logits, axis=-1) # [num_nodes, num_clusters]
        
        # 2. Coarse-grain node features
        # Create a block-diagonal assignment matrix for the batch
        batch_indices = jnp.repeat(
            jnp.arange(num_graphs), 
            graph.n_node, 
            total_repeat_length=num_nodes
        )
        
        # H_macro = S^T * H_micro (per graph)
        # We can use segment_sum to perform the contraction per graph
        # For each cluster k in graph b, H_macro[b, k] = sum_{i in graph b} S[i, k] * H_micro[i]
        coarse_nodes = []
        for k in range(self.num_clusters):
            weighted_nodes = node_feats * assignments[:, k:k+1]
            coarse_nodes_k = jraph.segment_sum(weighted_nodes, batch_indices, num_segments=num_graphs)
            coarse_nodes.append(coarse_nodes_k)
        
        # Stack to get [num_graphs, num_clusters, node_feat] and reshape to [num_graphs * num_clusters, node_feat]
        coarse_nodes = jnp.stack(coarse_nodes, axis=1).reshape(-1, node_feats.shape[-1])
        
        # 3. Coarse-grain adjacency (Simplified for batching)
        # In a real RG, we'd contract edges. For 80/20, we'll use a dense-to-sparse 
        # approach per graph or maintain a fully connected coarse graph for simplicity.
        # Here we'll just return the coarse nodes; the next layer can be a MLP or another GNN.
        # For a full GNN, we'd need to reconstruct the edges.
        
        # Reconstruct edges: for each graph, we'll assume a fully connected coarse graph for now
        # (This is standard in many set-pooling or graph-pooling architectures)
        c_n_node = jnp.full((num_graphs,), self.num_clusters)
        
        # Create fully connected edges for each coarse graph
        def get_fc_edges(n):
            adj = jnp.ones((n, n))
            return jnp.nonzero(adj)
        
        # For simplicity in batching, we'll return a graph with nodes only or FC edges
        # To keep it JIT-friendly, we use a fixed size
        single_fc_senders, single_fc_receivers = jnp.nonzero(
            jnp.ones((self.num_clusters, self.num_clusters)), 
            size=self.num_clusters**2
        )
        
        # Repeat for all graphs in batch
        batch_offset = jnp.arange(num_graphs)[:, None] * self.num_clusters
        c_senders = (single_fc_senders[None, :] + batch_offset).reshape(-1)
        c_receivers = (single_fc_receivers[None, :] + batch_offset).reshape(-1)
        
        coarse_graph = graph._replace(
            nodes=coarse_nodes,
            n_node=c_n_node,
            senders=c_senders,
            receivers=c_receivers,
            edges=jnp.ones((c_senders.shape[0], 1)),
            n_edge=jnp.full((num_graphs,), self.num_clusters**2)
        )
        
        return coarse_graph, assignments

class MSVIB(nn.Module):
    """Multi-Scale Variational Information Bottleneck with Hierarchical Pooling."""
    encoder_features: Iterable[int]
    latent_dim: int
    num_clusters_list: Iterable[int] # e.g., [16, 4] for multi-step
    output_dim: int = 1

    def setup(self):
        # Initial encoder for micro-states
        self.initial_encoder = GNNLayer(
            update_node_fn=lambda n, s, r, g: MLP(self.encoder_features)(n)
        )
        
        # Create multiple decimators and refinement encoders for multi-step renormalization
        self.decimators = [
            IterativeDecimator(num_clusters=nc) for nc in self.num_clusters_list
        ]
        
        # Encoders to refine features after each decimation
        self.refinement_encoders = [
            GNNLayer(update_node_fn=lambda n, s, r, g: MLP(self.encoder_features)(n))
            for _ in range(len(self.num_clusters_list))
        ]
        
        # Latent distributions for VIB
        self.fc_mu = nn.Dense(self.latent_dim)
        self.fc_logvar = nn.Dense(self.latent_dim)
        
        # Predictor head
        self.predictor = MLP([32, self.output_dim])

    def __call__(self, graph: jraph.GraphsTuple):
        # 1. Encode Micro-state
        h = self.initial_encoder(graph)
        
        # 2. Multi-step Renormalization (RG-like)
        current_graph = h
        all_assignments = []
        for i, decimator in enumerate(self.decimators):
            current_graph, assignments = decimator(current_graph)
            all_assignments.append(assignments)
            # Refine features after decimation
            current_graph = self.refinement_encoders[i](current_graph)
        
        # 3. Variational Bottleneck on Macro-state
        num_graphs = current_graph.n_node.shape[0]
        batch_indices = jnp.repeat(
            jnp.arange(num_graphs), 
            current_graph.n_node, 
            total_repeat_length=current_graph.nodes.shape[0]
        )
        
        # Global pooling of the final coarse representation
        macro_summary = jraph.segment_mean(
            current_graph.nodes, 
            batch_indices, 
            num_segments=num_graphs
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
        
        # 4. Predict outcome
        pred_y = self.predictor(z)
        
        return mu, logvar, pred_y, all_assignments, current_graph
