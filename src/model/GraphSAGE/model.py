import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn

from src.model.GraphSAGE.args import arg
from src.model.GraphSAGE.metric import metric

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


    def fit(self, train_loader, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr = arg.LR,
            weight_decay = arg.WEIGHT_DECAY
        )

        self.train()
        for epoch in range(epochs + 1):
            total_loss, val_loss, acc, val_acc = 0, 0, 0, 0

            for batch in train_loader:
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)

                loss = criterion(
                    out[batch.train_mask],
                    batch.y[batch.train_mask]
                )
                total_loss += loss

                acc += metric.accuracy(
                    out[batch.train_mask].argmax(dim = 1),
                    batch.y[batch.train_mask]
                )

                loss.backward()
                optimizer.step()

                val_loss += criterion(
                    out[batch.val_mask],
                    batch.y[batch.val_mask]
                )

                val_acc += metric.accuracy(
                    out[batch.val_mask].argmax(dim = 1),
                    batch.y[batch.val_mask]
                )

            if epoch % 20 == 0:
                print(
                    f'Epoch {epoch:>3} | Train Loss: {loss/len(train_loader):.3f} | Train Acc: {acc/len(train_loader)*100:>5.2f}%'
                    f'| Val Loss: {val_loss/len(train_loader):.2f} | Val Acc: {val_acc/len(train_loader)*100:.2f}%'
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