from torch import nn
class GAT(nn.Module) :
    def __init__(self, dim_h, dim_out) :
        super().__init__()
        import torch_geometric.nn as gnn

        self.conv = gnn.GATConv((-1, -1), dim_h, add_self_loops = False)
        self.linear = gnn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index) :
        h = self.conv(x, edge_index).relu()
        h = self.linear(h)
        return h