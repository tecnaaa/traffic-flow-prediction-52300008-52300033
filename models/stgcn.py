import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGLU(nn.Module):
    """Temporal gated convolution (Yu et al., 2018): (P + residual) * sigmoid(Q).

    Input/Output layout: (Batch, Channels, Nodes, Time). No padding, so the time
    axis shrinks by (kt - 1).
    """
    def __init__(self, c_in, c_out, kt):
        super().__init__()
        self.kt = kt
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(1, kt))
        self.align = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        residual = self.align(x)[..., self.kt - 1:]
        p, q = self.conv(x).chunk(2, dim=1)
        return (p + residual) * torch.sigmoid(q)


class SpatialGraphConv(nn.Module):
    """ReLU(W_nb (A X) + W_self X): giữ tín hiệu riêng của từng nút."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.theta_nb = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.theta_self = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)

    def forward(self, x, adj):
        agg = torch.einsum('nm,bcmt->bcnt', adj, x)
        return F.relu(self.theta_nb(agg) + self.theta_self(x))


class STConvBlock(nn.Module):
    """Temporal GLU -> Spatial graph conv -> Temporal GLU."""
    def __init__(self, c_in, c_spatial, c_out, kt):
        super().__init__()
        self.temporal1 = TemporalGLU(c_in, c_out, kt)
        self.spatial = SpatialGraphConv(c_out, c_spatial)
        self.temporal2 = TemporalGLU(c_spatial, c_out, kt)

    def forward(self, x, adj):
        x = self.temporal1(x)
        x = self.spatial(x, adj)
        return self.temporal2(x)


class STGCN(nn.Module):
    """STGCN baseline (simplified): two ST-Conv blocks + temporal output layer.

    x:   (Batch, TimeSteps, Nodes, Features)
    adj: (Nodes, Nodes) normalized adjacency
    out: (Batch, Nodes, out_dim)
    """
    def __init__(self, num_nodes, in_dim, out_dim, seq_len, kt=3):
        super().__init__()
        # Each ST block shortens the time axis by 2 * (kt - 1); two blocks -> 4 * (kt - 1)
        ko = seq_len - 4 * (kt - 1)
        if ko < 1:
            raise ValueError(f"seq_len={seq_len} is too short for kt={kt} "
                             f"(needs seq_len >= {4 * (kt - 1) + 1})")
        self.block1 = STConvBlock(in_dim, 16, 64, kt)
        self.block2 = STConvBlock(64, 16, 64, kt)
        self.output_temporal = TemporalGLU(64, 64, ko)  # collapses time axis to 1
        self.output_fc = nn.Conv2d(64, out_dim, kernel_size=1)

    def forward(self, x, adj):
        last = x[:, -1, :, 0:1]               
        x = x.permute(0, 3, 2, 1)
        x = self.block1(x, adj)
        x = self.block2(x, adj)
        x = self.output_temporal(x)
        x = self.output_fc(x).squeeze(-1)      
        return x.permute(0, 2, 1) + last       