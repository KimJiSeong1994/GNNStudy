import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.VanillaGNN.args import arg
from src.model.VanillaGNN.metrix import metric

class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x =  self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
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
            out = self(data.x)

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
                    f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%'
                )

    def test(self, data):
        self.eval()
        out = self(data.x)
        acc = metric.accuracy(
            out.argmax(dim = 1)[data.test_mask],
            data.y[data.test_mask]
        )

        return acc