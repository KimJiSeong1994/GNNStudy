from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

class GCN(nn.Module) :
    def __init__(self, num_features, num_classes, dim_h) :
        super().__init__()
        self.conv1 = gnn.GCNConv(num_features, dim_h)
        self.conv2 = gnn.GCNConv(dim_h, num_classes)

    def forward(self, x, edge_index) :
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p = 0.5, training = self.training)
        h = self.conv2(h, edge_index)
        return F.log_softmax(h, dim = 1)
