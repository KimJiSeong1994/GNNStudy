import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from src.model.GCN.NodeRegression.args import arg

class GCNRegreesion(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = gnn.GCNConv(dim_in, dim_h * 4)
        self.gcn2 = gnn.GCNConv(dim_h * 4, dim_h * 2)
        self.gcn3 = gnn.GCNConv(dim_h * 2, dim_h)
        self.linear = torch.nn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index):
        x =  self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p = 0.5, training = self.training)

        x =  self.gcn2(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p = 0.5, training = self.training)

        x = self.gcn3(x, edge_index)
        x = torch.relu(x)
        x = self.linear(x)
        return x

    def fit(self, data, epochs):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr = arg.LR,
            weight_decay = arg.WEIGHT_DECAY
        )

        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)

            loss = F.mse_loss(
                out.squeeze()[data.train_mask],
                data.y[data.train_mask].float()
            )

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = F.mse_loss(
                    out.squeeze()[data.val_mask],
                    data.y[data.val_mask].float()
                )

                print(
                    f'Epoch {epoch:>3} | Train Loss: {loss:.5f} | Val Loss: {val_loss:.5f}'
                )

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return F.mse_loss(
            out.squeeze()[data.test_mask],
            data.y[data.test_mask].float()
        )
