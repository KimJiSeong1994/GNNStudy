from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

class HeteroGNN(nn.Module) :
    def __init__(self, dim_h, dim_out, num_layers) :
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers) :
            conv = gnn.HeteroConv({
                ('host', 'to', 'flow') : gnn.SAGEConv((-1,-1), dim_h),
                ('flow', 'to', 'host'): gnn.SAGEConv((-1, -1), dim_h)
            }, aggr = 'sum')
            self.convs.append(conv)

        self.lin = nn.Linear(dim_h, dim_out)

    def forward(self, x_dict, edge_index_dict) :
        for conv in self.convs :
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        return self.lin(x_dict['flow'])