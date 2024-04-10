import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn

from src.model.GAT.args import arg
from src.model.GAT.metric import metric

class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads = 8):
        super().__init__()
        self.gat1 = gnn.GATv2Conv(dim_in, dim_h, heads = heads)
        self.gat2 = gnn.GATv2Conv(dim_h * heads, dim_out, heads = 1)

    def forward(self, x, edge_index) :
        x = F.dropout(x, p = 0.6, training = self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p = 0.6, training = self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim = 1)


    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr = arg.LR,
            weight_decay = arg.WEIGHT_DECAY
        )

        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)

            loss = criterion(
                out[data.train_mask],
                data.y[data.train_mask]
            )

            acc = metric.accuracy(
                out[data.train_mask].argmax(dim = 1),
                data.y[data.train_mask]
            )

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = criterion(
                    out[data.val_mask],
                    data.y[data.val_mask]
                )
                val_acc = metric.accuracy(
                    out[data.val_mask].argmax(dim = 1),
                    data.y[data.val_mask]
                )

                print(
                    f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}%'
                    f'| Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%'
                )

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = metric.accuracy(
            out.argmax(dim = 1)[data.test_mask],
            data.y[data.test_mask]
        )
        return acc