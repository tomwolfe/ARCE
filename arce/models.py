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
    use_gumbel: bool = True # Use Gumbel-Softmax for harder partitions

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, training: bool = True) -> Tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]:
        node_feats = graph.nodes
        num_nodes = node_feats.shape[0]
        num_graphs = graph.n_node.shape[0]
        
        # 1. Predict cluster assignment probabilities
        assignment_logits = MLP([32, self.num_clusters])(node_feats)
        
        if self.use_gumbel and training:
            # Gumbel-Softmax for more discrete-like assignments (Point 1)
            rng = self.make_rng('gumbel') if self.has_rng('gumbel') else jax.random.PRNGKey(0)
            
            # Manual Gumbel-Softmax implementation
            u = jax.random.uniform(rng, assignment_logits.shape, minval=1e-10, maxval=1.0)
            gumbels = -jnp.log(-jnp.log(u))
            assignments = nn.softmax((assignment_logits + gumbels) / 1.0, axis=-1)
        else:
            assignments = nn.softmax(assignment_logits, axis=-1)
        
        # 2. Coarse-grain node features
        batch_indices = jnp.repeat(
            jnp.arange(num_graphs), 
            graph.n_node, 
            total_repeat_length=num_nodes
        )
        
        coarse_nodes = []
        for k in range(self.num_clusters):
            weighted_nodes = node_feats * assignments[:, k:k+1]
            coarse_nodes_k = jraph.segment_sum(weighted_nodes, batch_indices, num_segments=num_graphs)
            coarse_nodes.append(coarse_nodes_k)
        
        coarse_nodes = jnp.stack(coarse_nodes, axis=1).reshape(-1, node_feats.shape[-1])
        
        # 3. Coarse-grain adjacency (A' = S^T * A * S)
        # We transition to a more sparse-friendly logic to avoid O(K^2) bottlenecks (Critique 1)
        s_senders = assignments[graph.senders] # [num_edges, num_clusters]
        s_receivers = assignments[graph.receivers] # [num_edges, num_clusters]
        edge_weights = graph.edges if graph.edges is not None else jnp.ones((graph.senders.shape[0], 1))
        
        edge_batch_indices = jnp.repeat(
            jnp.arange(num_graphs),
            graph.n_edge,
            total_repeat_length=graph.senders.shape[0]
        )

        def scan_body(carry, k):
            # Compute one row of the coarse adjacency matrix at a time to save memory
            # Use jax.lax.dynamic_slice to ensure static slice size for JIT
            col_k = jax.lax.dynamic_slice(s_senders, (0, k), (s_senders.shape[0], 1))
            contributions_k = col_k * s_receivers * edge_weights
            # row_k: [num_graphs, num_clusters]
            row_k = jraph.segment_sum(contributions_k, edge_batch_indices, num_segments=num_graphs)
            return carry, row_k

        _, coarse_adj_rows = jax.lax.scan(
            scan_body, 
            None, 
            jnp.arange(self.num_clusters)
        )
        # coarse_adj_dense: [num_graphs, num_clusters, num_clusters]
        coarse_adj_dense = jnp.transpose(coarse_adj_rows, (1, 0, 2))
        
        # 4. Convert to Sparse jraph representation (avoiding O(K^2) if possible)
        # If top_k_edges is set, we can keep the graph sparse.
        if self.top_k_edges is not None:
            # Only keep top K edges per coarse node
            # This is where we truly achieve O(K*top_k) instead of O(K^2)
            weights = coarse_adj_dense # [num_graphs, K, K]
            top_k_val, top_k_idx = jax.lax.top_k(weights, k=self.top_k_edges)
            
            # Reconstruct sparse senders/receivers from top_k
            batch_offset = jnp.arange(num_graphs)[:, None, None] * self.num_clusters
            # Properly broadcast senders to match top_k receivers
            senders_base = jnp.arange(self.num_clusters)[None, :, None]
            c_senders = (jnp.broadcast_to(senders_base, top_k_idx.shape) + batch_offset).reshape(-1)
            c_receivers = (top_k_idx + batch_offset).reshape(-1)
            c_edge_weights = top_k_val.reshape(-1, 1)
            n_edge_per_graph = self.num_clusters * self.top_k_edges
        else:
            # Full connectivity (FC) for the macro-nodes (Original logic but cleaned)
            single_fc_senders, single_fc_receivers = jnp.nonzero(
                jnp.ones((self.num_clusters, self.num_clusters)), 
                size=self.num_clusters**2
            )
            
            batch_offset = jnp.arange(num_graphs)[:, None] * self.num_clusters
            c_senders = (single_fc_senders[None, :] + batch_offset).reshape(-1)
            c_receivers = (single_fc_receivers[None, :] + batch_offset).reshape(-1)
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
        
        return coarse_graph, assignments, coarse_adj_dense

class MSVIB(nn.Module):
    """Multi-Scale Variational Information Bottleneck with Hierarchical Pooling."""
    encoder_features: Iterable[int]
    latent_dim: int
    num_clusters_list: Iterable[int] # e.g., [16, 4] for multi-step
    output_dim: int = 1
    top_k_edges: Optional[int] = None
    use_gumbel: bool = True

    def setup(self):
        # Initial encoder for micro-states
        self.initial_encoder = GNNLayer(
            update_node_fn=lambda n, s, r, g: MLP(self.encoder_features)(n)
        )
        
        # Create multiple decimators and refinement encoders for multi-step renormalization
        self.decimators = [
            IterativeDecimator(num_clusters=nc, top_k_edges=self.top_k_edges, use_gumbel=self.use_gumbel) 
            for nc in self.num_clusters_list
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
        
        # Decoder for reconstruction (NEW)
        # Reconstructs coarse node features from latent z
        self.decoder = MLP(list(self.encoder_features[::-1]) + [self.encoder_features[-1] * self.num_clusters_list[-1]])

    def __call__(self, graph: jraph.GraphsTuple, training: bool = True):
        # 1. Encode Micro-state
        h_micro = self.initial_encoder(graph).nodes
        h = graph._replace(nodes=h_micro)
        
        # 2. Multi-step Renormalization (RG-like)
        current_graph = h
        all_assignments = []
        all_coarse_adjs = []
        n_node_history = [graph.n_node]
        for i, decimator in enumerate(self.decimators):
            current_graph, assignments, coarse_adj = decimator(current_graph, training=training)
            all_assignments.append(assignments)
            all_coarse_adjs.append(coarse_adj)
            n_node_history.append(current_graph.n_node)
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
        if training:
            rng = self.make_rng('vmap_rng') if self.has_rng('vmap_rng') else jax.random.PRNGKey(0)
            eps = jax.random.normal(rng, mu.shape)
        else:
            eps = jnp.zeros_like(mu)
        z = mu + eps * std
        
        # 4. Predict outcome
        pred_y = self.predictor(z)
        
        # 5. Reconstruction (NEW)
        # Reconstruct final coarse nodes from z
        h_final_coarse_recon = self.decoder(z) # [num_graphs, num_clusters_final * feat]
        h_final_coarse_recon = h_final_coarse_recon.reshape(num_graphs, self.num_clusters_list[-1], -1)
        
        # Back-project through the hierarchy
        curr_h_recon = h_final_coarse_recon
        for i in reversed(range(len(all_assignments))):
            S = all_assignments[i] # [total_nodes_prev, num_clusters_curr]
            
            # Map batch indices to nodes
            prev_n_node = n_node_history[i]
            prev_batch_indices = jnp.repeat(
                jnp.arange(num_graphs), 
                prev_n_node, 
                total_repeat_length=S.shape[0]
            )
            
            # H_prev[i] = sum_k S[i, k] * H_curr[batch(i), k]
            # Using gather to get H_curr for the correct graph
            h_curr_per_node = curr_h_recon[prev_batch_indices] # [total_nodes_prev, num_clusters_curr, feat]
            
            # Weighted sum over clusters
            # S[:, :, None] * h_curr_per_node
            curr_h_recon = jnp.sum(S[..., None] * h_curr_per_node, axis=1) # [total_nodes_prev, feat]
            
            # If not the last step, reshape for next iteration
            if i > 0:
                num_clusters_prev = self.num_clusters_list[i-1]
                curr_h_recon = curr_h_recon.reshape(num_graphs, num_clusters_prev, -1)
        
        recon_micro = curr_h_recon # [total_nodes_micro, feat]
        
        return mu, logvar, pred_y, all_assignments, current_graph, recon_micro, h_micro, all_coarse_adjs
