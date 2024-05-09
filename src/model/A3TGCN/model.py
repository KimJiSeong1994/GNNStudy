from torch import nn
from torch_geometric_temporal import nn as TGNN

class A3TGCN(nn.Module) :
    def __init__(self, dim_in, periods) :
        super().__init__()
        self.tgnn = TGNN.A3TGCN(in_channels = dim_in, out_channels = 32, periods = periods)
        self.linear = nn.Linear(32, periods)

    def forward(self, x, edge_index, edge_attr) :
        h = self.tgnn(x, edge_index, edge_attr).relu()
        h = self.linear(h)
        return h