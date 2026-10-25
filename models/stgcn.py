import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x: (Batch, Nodes, Features)
        # adj: (Nodes, Nodes) 
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return F.relu(output)

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, seq_len):
        super(STGCN, self).__init__()
        self.gcn1 = GraphConv(in_dim, 64)
        self.gcn2 = GraphConv(64, 32)
        # Fully connected layer 
        self.fc = nn.Linear(32 * seq_len, out_dim)

    def forward(self, x, adj):
        # x shape: (Batch, TimeSteps, Nodes, Features)
        batch_size, time_steps, num_nodes, num_features = x.shape
        
        out = []
        for t in range(time_steps):
            gcn_out = self.gcn1(x[:, t, :, :], adj)
            gcn_out = self.gcn2(gcn_out, adj)
            out.append(gcn_out.unsqueeze(1))
            
        # (Batch, TimeSteps, Nodes, 32)
        out = torch.cat(out, dim=1) 
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        out = self.fc(out)
        return out