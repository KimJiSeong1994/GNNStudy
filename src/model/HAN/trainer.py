if __name__ == '__main__' :
    import torch
    import torch.nn.functional as F
    import torch_geometric.nn as gnn

    from src.model.HAN.model import HAN
    from src.model.HAN.GetData import get
    from src.model.HAN.args import arg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = get()
    dataset, data = loader['dataset'], loader['data']
    data['conference'].x = torch.zeros(20, 1)

    model = HAN(dim_in = -1, dim_out = 4, metapath = data.metadata())
    optimizer = torch.optim.Adam(model.parameters(), lr = arg.LEARINGRATE, weight_decay = arg.WEIGHT_DECAY)
    data, model = data.to(device), model.to(device)

    @torch.no_grad()
    def test(mask) :
        model.eval()
        pred = model(data.x_dict, data.edge_index_dict).argmax(dim = -1)
        acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
        return float(acc)

    for epoch in range(arg.EPOCH + 1) :
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        mask = data['author'].train_mask

        loss = F.cross_entropy(out[mask], data['author'].y[mask])
        loss.backward()
        optimizer.step()

        if epoch % 20  == 0 :
            train_acc = test(data['author'].train_mask)
            val_acc = test(data['author'].val_mask)
            print(f'Epoch: {epoch:>3} | Train loss: {loss:.4f} | Tain Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc*100:.2f}%')

    test_acc = test(data['author'].test_mask)
    print(f'Test Acc: {test_acc * 100:.2f}%')