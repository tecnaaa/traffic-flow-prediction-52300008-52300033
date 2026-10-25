import torch
import torch.nn as nn
import torch.nn.functional as F

class DualGraphConv(nn.Module):
    """
    Dual-Graph Convolution combining Physical Adjacency and Data-driven Hyperedge/Correlation Matrix.
    Core contribution of the proposed HGMWN model.
    """
    def __init__(self, in_features, out_features):
        super(DualGraphConv, self).__init__()
        self.weight_physical = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_data = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.alpha = nn.Parameter(torch.FloatTensor(1)) # Learnable fusion weight
        
        nn.init.xavier_uniform_(self.weight_physical)
        nn.init.xavier_uniform_(self.weight_data)
        nn.init.constant_(self.alpha, 0.5)

    def forward(self, x, adj_physical, adj_data):
        # x: (Batch, Nodes, Features)
        # adj_physical: Physical spatial graph (Nodes, Nodes)
        # adj_data: Data-driven correlation graph (Nodes, Nodes)
        
        support_phys = torch.matmul(x, self.weight_physical)
        out_phys = torch.matmul(adj_physical, support_phys)
        
        support_data = torch.matmul(x, self.weight_data)
        out_data = torch.matmul(adj_data, support_data)
        
        # Dual-graph adaptive fusion
        alpha = torch.sigmoid(self.alpha)
        output = alpha * out_phys + (1 - alpha) * out_data
        return F.relu(output)

class HGMWN(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, seq_len):
        super(HGMWN, self).__init__()
        self.conv1 = DualGraphConv(in_dim, 64)
        self.conv2 = DualGraphConv(64, 32)
        self.fc = nn.Linear(32 * seq_len, out_dim)

    def forward(self, x, adj_physical, adj_data):
        # x shape: (Batch, TimeSteps, Nodes, Features)
        batch_size, time_steps, num_nodes, num_features = x.shape
        
        out = []
        for t in range(time_steps):
            h = self.conv1(x[:, t, :, :], adj_physical, adj_data)
            h = self.conv2(h, adj_physical, adj_data)
            out.append(h.unsqueeze(1))
            
        out = torch.cat(out, dim=1) # (Batch, TimeSteps, Nodes, 32)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        out = self.fc(out)
        return out