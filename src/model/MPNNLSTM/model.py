from torch import nn
import torch_geometric_temporal.nn as TGNN

class MPNNLSTM(nn.Module) :
    def __init__(self, dim_in, dim_h, num_nodes) :
        super().__init__()
        self.recurrent = TGNN.MPNNLSTM(dim_in, dim_h, num_nodes, 1, 0.5)
        self.dropout = nn.Dropout(.5)
        self.linear = nn.Linear(2 * dim_h + dim_in, 1)

    def forward(self, x, edge_index, edge_weight) :
        h = self.recurrent(x, edge_index, edge_weight).relu()
        h = self.dropout(h)
        h = self.linear(h).tanh()
        return h