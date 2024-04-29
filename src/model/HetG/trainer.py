if __name__ == '__main__' :
    import torch
    import torch.nn.functional as F
    import torch_geometric.nn as gnn

    from src.model.HetG.GetData import get
    from src.model.HetG.args import arg

    loader = get()
    dataset, data = loader['dataset'], loader['data']

    model = gnn.GAT(
        in_channels = -1,
        hidden_channels = arg.HIDDEN_SIZE,
        out_channels = arg.OUT_SIZE,
        num_layers = arg.NUM_LAYER
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr = arg.LEARNINGRATE, weight_decay = arg.WEIGHT_DECAY)
    data, model = data.to(device), model.to(device)

    @torch.no_grad()
    def test(mask) :
        model.eval()
        pred = model(
            data.x_dict['author'],
            data.edge_index_dict[('author', 'metapath_0', 'author')]
        ).argmax(dim = -1)
        return float((pred[mask] == data['author'].y[mask]).sum() / mask.sum())

    for epoch in range(arg.EPOCH + 1) :
        model.train()
        optimizer.zero_grad()
        out = model(
            data.x_dict['author'],
            data.edge_index_dict[('author', 'metapath_0', 'author')],
        )

        mask = data['author'].train_mask
        loss = F.cross_entropy(
            out[mask],
            data['author'].y[mask]
        )

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 :
            train_acc = test(data['author'].train_mask)
            val_acc = test(data['author'].val_mask)
            print(f'Epoch: {epoch:>3} | Train Lose:{loss:.4f} | Train Acc: {train_acc *100:.2f}% | Val Acc: {val_acc*100:.2f}%')

    test_acc = test(data['author'].test_mask)
    print(f'Test Acc: {test_acc*100:.2f}%')