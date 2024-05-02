import torch
from torch import nn
import torch_geometric_temporal.nn as TGNN

class TemporalGNN(nn.Module) :
    def __init__(self, node_count, dim_in) :
        super().__init__()
        self.recurrent = TGNN.EvolveGCNH(node_count, dim_in)
        self.linear = torch.nn.Linear(dim_in, 1)

    def forward(self, x, edge_index, edge_weight) :
        h = self.recurrent(x, edge_index, edge_weight).relu()
        h = self.linear(h)
        return h
