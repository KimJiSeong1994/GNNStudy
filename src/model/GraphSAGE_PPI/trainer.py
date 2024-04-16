if __name__ == '__main__' :
    import torch
    from src.model.GraphSAGE_PPI import GetData
    from src.model.GraphSAGE_PPI.args import arg
    from torch_geometric.nn import GraphSAGE

    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader
    from torch_geometric.loader import NeighborLoader
    from sklearn.metrics import f1_score
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloder = GetData.get()
    train_data = Batch.from_data_list(dataloder['train'])
    loader = NeighborLoader(
        train_data,
        batch_size = arg.BATCH,
        num_neighbors = arg.NUM_NEIGHTBORS,
        num_workers = 2,
        persistent_workers = True,
        shuffle = True
    )

    train_loader = DataLoader(dataloder['train'], batch_size = 2)
    val_loader = DataLoader(dataloder['val'], batch_size = 2)
    test_loader = DataLoader(dataloder['test'], batch_size = 2)

    model = GraphSAGE(
        in_channels = train_data.num_features,
        hidden_channels = arg.HIDDEN_SIZE * 2,
        num_layers = 2,
        out_channels = dataloder['train'].num_classes
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = arg.LR)

    def fit():
        model.train()
        total_loss = 0

        for data  in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            loss.backward()
            optimizer.step()

        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()
        data = next(iter(loader))
        out = model(data.x.to(device), data.edge_index.to(device))

        preds = (out > 9).float().cpu()
        y, pred = data.y.numpy(), preds.numpy()
        return f1_score(y, pred, average = 'micro') if pred.sum() > 0 else 0

    for epoch in range(arg.EPOCHS):
        loss = fit()
        val_f1 = test(val_loader)
        if epoch % 50 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Val F1 score: {val_f1:.4f}')

    print(f'Test F1 socre : {test(test_loader):.4f}')