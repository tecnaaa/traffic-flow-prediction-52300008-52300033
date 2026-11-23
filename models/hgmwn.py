import torch
import torch.nn as nn
import torch.nn.functional as F


class DualGraphConv(nn.Module):
    """Dual-graph convolution: physical graph + data-driven hypergraph operator.

    mode:
        'phys' : physical graph only            (ablation)
        'data' : data-driven hyperedge only     (ablation)
        'dual' : both graphs, fused by a learnable weight alpha (proposed)

    Only the parameters needed by the chosen mode are created, so parameter
    counts stay honest when comparing ablation variants.
    """
    MODES = ('phys', 'data', 'dual')

    def __init__(self, in_features, out_features, mode='dual'):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")
        self.mode = mode

        if mode in ('phys', 'dual'):
            self.weight_physical = nn.Parameter(torch.empty(in_features, out_features))
            nn.init.xavier_uniform_(self.weight_physical)
        if mode in ('data', 'dual'):
            self.weight_data = nn.Parameter(torch.empty(in_features, out_features))
            nn.init.xavier_uniform_(self.weight_data)
        if mode == 'dual':
            # Learnable fusion weight (sigmoid applied in forward)
            self.alpha = nn.Parameter(torch.full((1,), 0.5))

    def forward(self, x, adj_physical, adj_data):
        # x: (..., Nodes, Features); adjacency matrices: (Nodes, Nodes)
        # torch.matmul broadcasts the adjacency over all leading dims
        # (batch, time), so no Python loop over time steps is needed.
        if self.mode == 'phys':
            out = torch.matmul(adj_physical, torch.matmul(x, self.weight_physical))
        elif self.mode == 'data':
            out = torch.matmul(adj_data, torch.matmul(x, self.weight_data))
        else:
            out_phys = torch.matmul(adj_physical, torch.matmul(x, self.weight_physical))
            out_data = torch.matmul(adj_data, torch.matmul(x, self.weight_data))
            alpha = torch.sigmoid(self.alpha)
            out = alpha * out_phys + (1 - alpha) * out_data
        return F.relu(out)

    def alpha_value(self):
        """Weight given to the physical graph (None if not in dual mode)."""
        if self.mode != 'dual':
            return None
        return torch.sigmoid(self.alpha).item()


class HGMWN(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, seq_len, mode='dual'):
        super().__init__()
        self.mode = mode
        self.conv1 = DualGraphConv(in_dim, 64, mode=mode)
        self.conv2 = DualGraphConv(64, 32, mode=mode)
        self.fc = nn.Linear(32 * seq_len, out_dim)

    def forward(self, x, adj_physical, adj_data):
        # x: (Batch, TimeSteps, Nodes, Features)
        batch_size, _, num_nodes, _ = x.shape

        h = self.conv1(x, adj_physical, adj_data)   # (B, T, N, 64)
        h = self.conv2(h, adj_physical, adj_data)   # (B, T, N, 32)

        h = h.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)  # (B, N, T*32)
        return self.fc(h)                                              # (B, N, out_dim)

    def alphas(self):
        """Fusion weights (physical-graph share) of each layer, dual mode only."""
        values = [self.conv1.alpha_value(), self.conv2.alpha_value()]
        return [v for v in values if v is not None]
