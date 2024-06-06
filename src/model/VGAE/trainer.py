if __name__ == '__main__' :
    import torch
    import torch_geometric.nn as gnn

    from src.model.VGAE import model
    from src.model.VGAE.args import arg
    from src.model.VGAE.GetData import get
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloder = get()
    dataset = dataloder['data']
    train_set, val_set, test_set = dataloder['train'], dataloder['val'], dataloder['test']

    model = gnn.VGAE(model.Encoder(dataset.num_features, arg.HIDDEN_SIZE)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = arg.LR)

    def train() :
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_set.x, train_set.edge_index)
        loss = model.recon_loss(z, train_set.pos_edge_label_index) + (1/ train_set.num_nodes) * model.kl_loss()

        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(data) :
        model.eval()
        z = model.encode(data.x, data.edge_index)
        return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    for epoch in range(arg.EPOCHS + 1) :
        loss = train()
        val_auc, val_ap = test(val_set)
        if epoch % 50 == 0 :
            print(
                f'Epoch {epoch:>2} | Loss : {loss:.4f} | Val AUC : {val_auc:.4f} | Val AP : {val_ap:.4f}'
            )

    test_auc, test_ap = test(test_set)
    print(f'Test AUC: {test_auc:.4f} | Test AP {test_ap:.4f}')

    z  = model.encode(test_set.x, test_set.edge_index)
    Ahat = torch.sigmoid(z @ z.T)