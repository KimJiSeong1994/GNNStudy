if __name__ == '__main__' :
    import torch
    from src.model.EvolveGCN.GetData import get
    from src.model.EvolveGCN.model import TemporalGNN

    loader = get()
    dataset, train_set, test_set = loader['dataset'], loader['train'], loader['test']

    model = TemporalGNN(
        node_count = dataset[0].x.shape[0],
        dim_in = dataset[0].x.shape[1]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=.01)

    model.train()
    for epoch in range(50) :
        for i, snapshot in enumerate(train_set) : # snapshot : Temporal sequence dim
            y_pred = model(
                snapshot.x,
                snapshot.edge_index,
                snapshot.edge_attr
            )

            loss = torch.mean((y_pred - snapshot.y) ** 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 5 == 0 :
            print(f'Epoch: {epoch}, Train Loss: {loss:.4f}')

    loss = 0
    model.eval()
    for i, snapshot in enumerate(test_set) :
        y_pred = model(
            snapshot.x,
            snapshot.edge_index,
            snapshot.edge_attr
        )

        mse = torch.mean((y_pred - snapshot.y) ** 2)
        loss += mse

    loss = loss/(i+1)
    print(f'MSE = {loss.item():.4f}')