if __name__ == '__main__' :
    import torch
    from src.model.MPNNLSTM.GetData import get
    from src.model.MPNNLSTM.model import MPNNLSTM

    loader = get()
    dataset, train_set, test_set = loader['data'], loader['train'], loader['test']

    model = MPNNLSTM(
        dataset[0].x.shape[1],
        64,
        dataset[0].x.shape[0]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    model.train()
    for epoch in range(100) :
        loss = 0
        for i, snapshot in enumerate(train_set) :
            y_pred = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            loss += torch.mean((y_pred - snapshot.y) ** 2)

        loss /= (i+1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    loss = 0
    for i, snapshot in enumerate(test_set) :
        y_pred = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        mse = torch.mean((y_pred - snapshot.y) ** 2)
        loss += mse

    loss /= (i+1)
    print(f'MSE : {loss.item():.4f}')
