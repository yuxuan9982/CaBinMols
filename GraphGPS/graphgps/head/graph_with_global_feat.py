import torch
import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head
from torch_geometric.graphgym.models.layer import new_layer_config, MLP


@register_head('graph_with_global_feat')
class GraphHeadWithGlobalFeat(nn.Module):
    """
    Graph prediction head that supports graph-level input features (data.u).
    This head concatenates the pooled graph representation with graph-level features
    before making predictions.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        
        # Check if graph-level features are available
        # If data.u exists, we need to account for its dimension
        # For now, we assume data.u has dimension 1 (single scalar feature)
        # This can be made configurable if needed
        self.has_global_feat = True  # Assume we always have global features
        global_feat_dim = 1  # dE_triplet is a scalar
        
        # Combine graph embedding with global features
        combined_dim = dim_in + global_feat_dim
        
        # Use MLP for the prediction layers
        self.layers = MLP(
            new_layer_config(combined_dim, dim_out, cfg.gnn.layers_post_mp,
                           has_act=False, has_bias=True, cfg=cfg))
        
        # Optional: normalization layer
        # if cfg.gnn.layers_post_mp > 0:
        #     self.ln = torch.nn.LayerNorm(dim_in)
        # else:
        self.ln = None

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        # Get graph-level embedding via pooling
        if self.ln is not None:
            x = self.ln(batch.x)
        else:
            x = batch.x
        graph_emb = self.pooling_fun(x, batch.batch)
        
        # Concatenate with graph-level features (data.u)
        if hasattr(batch, 'u') and batch.u is not None:
            # batch.u shape: [num_graphs, global_feat_dim]
            # Ensure it matches the batch size
            if batch.u.dim() == 1:
                batch.u = batch.u.unsqueeze(1)  # [num_graphs, 1]
            # Convert batch.u to the same dtype as graph_emb to avoid type mismatch
            # (data.u is created as double, but model weights are float)
            batch.u = batch.u.to(dtype=graph_emb.dtype)
            graph_emb = torch.cat([graph_emb, batch.u], dim=1)
        else:
            # If no global features, use zero padding or skip
            # For backward compatibility, we can use zeros
            num_graphs = graph_emb.shape[0]
            zero_global_feat = torch.zeros(num_graphs, 1, device=graph_emb.device, dtype=graph_emb.dtype)
            graph_emb = torch.cat([graph_emb, zero_global_feat], dim=1)
        
        # Make prediction
        graph_emb = self.layers(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label
