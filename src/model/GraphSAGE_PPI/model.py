import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn

class GraphSAGE(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = gnn.SAGEConv(dim_in, dim_h)
        self.sage2 = gnn.SAGEConv(dim_h, dim_out)

    def forward(self, x, edge_index) :
        x = self.sage1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p = 0.5, training = self.training)

        x = self.sage2(x, edge_index)
        return F.log_softmax(x, dim = 1)
