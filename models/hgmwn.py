import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_sparsify(adj, k):
    """Keep the k largest entries per row (diagonal excluded); zero out the rest.

    adj: numpy array (N, N). Returns an unnormalized numpy array (N, N).
    """
    a = adj.astype(np.float32).copy()
    np.fill_diagonal(a, 0.0)
    if k <= 0 or k >= a.shape[0] - 1:
        return a
    out = np.zeros_like(a)
    idx = np.argpartition(-a, k - 1, axis=1)[:, :k]
    rows = np.arange(a.shape[0])[:, None]
    out[rows, idx] = a[rows, idx]
    return out


class TemporalGLU(nn.Module):
    """Gated temporal convolution: (P + skip) * sigmoid(Q).

    Uses 'same' padding, so the time axis keeps its length. The dilation
    enlarges the receptive field without adding parameters.

    Input/Output layout: (B, T, N, C).
    """
    def __init__(self, c_in, c_out, kernel=3, dilation=1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(1, kernel),
                              padding=(0, pad), dilation=(1, dilation))
        self.align = (nn.Conv2d(c_in, c_out, kernel_size=1)
                      if c_in != c_out else nn.Identity())

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)                  # (B, C, N, T)
        p, q = self.conv(x).chunk(2, dim=1)
        out = (p + self.align(x)) * torch.sigmoid(q)
        return out.permute(0, 3, 2, 1)             # (B, T, N, C)


class DualGraphConv(nn.Module):
    """ReLU( A_phys X W_p  [+ A_data X W_d]  +  X W_self ).

    mode:
        'phys' : physical graph + skip connection
        'data' : data-driven hyperedge operator + skip connection
        'dual' : both graphs, fused by a learnable weight alpha

    The skip connection (W_self) exists in EVERY mode so that the ablation is
    fair: variants differ only in the graph propagation part.
    """
    MODES = ('phys', 'data', 'dual')

    def __init__(self, in_features, out_features, mode='dual'):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")
        self.mode = mode
        self.w_self = nn.Linear(in_features, out_features)
        if mode in ('phys', 'dual'):
            self.w_phys = nn.Linear(in_features, out_features, bias=False)
        if mode in ('data', 'dual'):
            self.w_data = nn.Linear(in_features, out_features, bias=False)
        if mode == 'dual':
            # Learnable fusion weight; sigmoid(0) = 0.5 at initialization
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj_physical, adj_data):
        # x: (B, T, N, F); adj: (N, N) -> matmul broadcasts over (B, T)
        if self.mode == 'phys':
            agg = self.w_phys(torch.matmul(adj_physical, x))
        elif self.mode == 'data':
            agg = self.w_data(torch.matmul(adj_data, x))
        else:
            a = torch.sigmoid(self.alpha)
            agg = (a * self.w_phys(torch.matmul(adj_physical, x))
                   + (1 - a) * self.w_data(torch.matmul(adj_data, x)))
        return F.relu(agg + self.w_self(x))

    def alpha_value(self):
        """Weight given to the physical graph (None if not in dual mode)."""
        if self.mode != 'dual':
            return None
        return torch.sigmoid(self.alpha).item()


class HGMWN(nn.Module):
    """Temporal GLU -> DualGraphConv -> Temporal GLU -> DualGraphConv -> head.

    The temporal part is identical in all modes; only the graph propagation
    differs, so phys/data/dual ablations stay fair. The head predicts a delta
    with respect to the last observed (standardized) value.
    """
    def __init__(self, num_nodes, in_dim, out_dim, seq_len, mode='dual', hidden=64):
        super().__init__()
        self.mode = mode
        self.tconv1 = TemporalGLU(in_dim, hidden, kernel=3, dilation=1)
        self.gconv1 = DualGraphConv(hidden, hidden, mode=mode)
        self.tconv2 = TemporalGLU(hidden, hidden, kernel=3, dilation=2)
        self.gconv2 = DualGraphConv(hidden, hidden, mode=mode)
        self.fc = nn.Linear(hidden * seq_len, out_dim)

    def forward(self, x, adj_physical, adj_data):
        # x: (B, T, N, F); channel 0 is the standardized flow
        batch_size, _, num_nodes, _ = x.shape
        last = x[:, -1, :, 0:1]                                   # (B, N, 1) last observed value

        h = self.tconv1(x)                                        # (B, T, N, H)
        h = self.gconv1(h, adj_physical, adj_data)                # (B, T, N, H)
        h = self.tconv2(h)                                        # (B, T, N, H)
        h = self.gconv2(h, adj_physical, adj_data)                # (B, T, N, H)

        h = h.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)  # (B, N, T*H)
        return self.fc(h) + last                                  # predict delta w.r.t. last value

    def alphas(self):
        """Fusion weights (physical-graph share) of each layer, dual mode only."""
        values = [self.gconv1.alpha_value(), self.gconv2.alpha_value()]
        return [v for v in values if v is not None]