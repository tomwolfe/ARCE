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
    top_k_edges: Optional[int] = None # If set, keeps only top K edges per cluster

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
        
        # 3. Coarse-grain adjacency (A' = S^T * A * S)
        # We need to aggregate micro-edges based on assignments
        # Micro-edges: senders [E], receivers [E]
        # assignments: [num_nodes, num_clusters]
        
        # S[i, k] is the probability node i belongs to cluster k
        # A'_{kl} = sum_{i, j} S[i, k] * A_{ij} * S[j, l]
        # For each micro-edge (i, j), it contributes S[i, k] * S[j, l] to A'_{kl}
        
        s_senders = assignments[graph.senders] # [num_edges, num_clusters]
        s_receivers = assignments[graph.receivers] # [num_edges, num_clusters]
        
        # Optimized edge aggregation to avoid O(E * K^2) memory bottleneck
        # We want: coarse_adj_dense[b, k, l] = sum_{e in graph b} S[senders[e], k] * S[receivers[e], l]
        # We can compute this by iterating over k to keep intermediate tensors at O(E * K)
        
        edge_batch_indices = jnp.repeat(
            jnp.arange(num_graphs),
            graph.n_edge,
            total_repeat_length=graph.senders.shape[0]
        )

        # Optimization: Use jax.lax.scan instead of vmap to keep memory footprint at O(E * K)
        # instead of O(K * E * K) which would happen if vmap parallelizes over K.
        def scan_body(carry, k):
            sender_k = jnp.take(s_senders, k, axis=1)[:, None]
            contributions_k = sender_k * s_receivers
            row_k = jraph.segment_sum(contributions_k, edge_batch_indices, num_segments=num_graphs)
            return carry, row_k

        _, coarse_adj_rows = jax.lax.scan(
            scan_body, 
            None, 
            jnp.arange(self.num_clusters)
        )
        # coarse_adj_rows is [num_clusters, num_graphs, num_clusters]
        # Transpose to [num_graphs, num_clusters, num_clusters]
        coarse_adj_dense = jnp.transpose(coarse_adj_rows, (1, 0, 2))
        
        # 4. Convert dense coarse adjacency back to sparse
        if self.top_k_edges is not None:
            # For each cluster, keep only top_k outgoing edges
            k = min(self.top_k_edges, self.num_clusters)
            top_vals, top_indices = jax.lax.top_k(coarse_adj_dense, k)
            
            # Reconstruct sparse representation
            single_fc_senders = jnp.repeat(jnp.arange(self.num_clusters), k)
            single_fc_receivers = top_indices.reshape(num_graphs, -1)
            
            batch_offset = jnp.arange(num_graphs)[:, None] * self.num_clusters
            c_senders = (single_fc_senders[None, :] + batch_offset).reshape(-1)
            # Use advanced indexing to get receivers correctly for the batch
            c_receivers = (single_fc_receivers + batch_offset).reshape(-1)
            
            c_edge_weights = top_vals.reshape(-1, 1)
            n_edge_per_graph = self.num_clusters * k
        else:
            # Full connectivity (FC)
            single_fc_senders, single_fc_receivers = jnp.nonzero(
                jnp.ones((self.num_clusters, self.num_clusters)), 
                size=self.num_clusters**2
            )
            
            batch_offset = jnp.arange(num_graphs)[:, None] * self.num_clusters
            c_senders = (single_fc_senders[None, :] + batch_offset).reshape(-1)
            c_receivers = (single_fc_receivers[None, :] + batch_offset).reshape(-1)
            
            # Weights for these edges
            c_edge_weights = coarse_adj_dense[:, single_fc_senders, single_fc_receivers].reshape(-1, 1)
            n_edge_per_graph = self.num_clusters**2
        
        c_n_node = jnp.full((num_graphs,), self.num_clusters)
        
        coarse_graph = graph._replace(
            nodes=coarse_nodes,
            n_node=c_n_node,
            senders=c_senders,
            receivers=c_receivers,
            edges=c_edge_weights,
            n_edge=jnp.full((num_graphs,), n_edge_per_graph)
        )
        
        return coarse_graph, assignments

class MSVIB(nn.Module):
    """Multi-Scale Variational Information Bottleneck with Hierarchical Pooling."""
    encoder_features: Iterable[int]
    latent_dim: int
    num_clusters_list: Iterable[int] # e.g., [16, 4] for multi-step
    output_dim: int = 1
    top_k_edges: Optional[int] = None

    def setup(self):
        # Initial encoder for micro-states
        self.initial_encoder = GNNLayer(
            update_node_fn=lambda n, s, r, g: MLP(self.encoder_features)(n)
        )
        
        # Create multiple decimators and refinement encoders for multi-step renormalization
        self.decimators = [
            IterativeDecimator(num_clusters=nc, top_k_edges=self.top_k_edges) for nc in self.num_clusters_list
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
