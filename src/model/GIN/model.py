import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
torch.manual_seed(42)

class GIN(nn.Module) :
    def __init__(self, dim_h, node_feature, num_classes) :
        super(GIN, self).__init__()
        self.conv1 = gnn.GINConv(
            nn.Sequential(
                nn.Linear(node_feature, dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU()
            )
        )

        self.conv2 = gnn.GINConv(
            nn.Sequential(
                nn.Linear(dim_h, dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU()
            )
        )

        self.conv3 = gnn.GINConv(
            nn.Sequential(
                nn.Linear(dim_h, dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU()
            )
        )

        self.lin1 = nn.Linear(dim_h * 3, dim_h * 3)
        self.lin2 = nn.Linear(dim_h * 3, num_classes)

    def forward(self, x, edge_index, batch) :
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        h1 = gnn.global_add_pool(h1, batch)
        h2 = gnn.global_add_pool(h2, batch)
        h3 = gnn.global_add_pool(h3, batch)

        h = torch.cat((h1, h2, h3), dim = 1)
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p = 0.5, training = self.training)
        h = self.lin2(h)
        return F.log_softmax(h, dim = 1)