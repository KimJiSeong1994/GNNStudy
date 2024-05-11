if __name__ == '__main__' :
    import torch
    import numpy as np
    from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split

    from src.model.A3TGCN.GetData import get
    from src.model.A3TGCN.Utils import utils
    from src.model.A3TGCN.model import A3TGCN

    loader = get()
    speed, dist = loader['speed'], loader['dist']

    adj = utils.compute_adj(dist)
    speed_norm = utils.zscore(speed, speed.mean(axis = 0), speed.std(axis = 0))

    N = speed_norm.shape[0]
    lags = 24
    horizon = 48
    xs, ys = [], []
    for i in range(lags, (N - horizon)) :
        xs.append(speed_norm.to_numpy()[i-lags :i].T)
        ys.append(speed_norm.to_numpy()[i+horizon-1])

    edge_index = (np.array(adj) > 0).nonzero()
    edge_weight = adj[adj > 0]

    dataset = StaticGraphTemporalSignal(edge_index, adj[adj > 0], xs, ys)
    tarin_dataset, test_dataset = temporal_signal_split(dataset, train_ratio = .8)

    model = A3TGCN(lags, 1).to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-3)

    model.train()
    for epoch in range(30) :
        loss, step = 0, 0

        for i, snapshot in enumerate(tarin_dataset) :
            y_pred = model(
                snapshot.x.unsqueeze(2),
                snapshot.edge_index,
                snapshot.edge_attr
            )

            loss += torch.mean((y_pred - snapshot.y) ** 2)
            step += 1

        loss = loss / (step + 1)
        loss.backward()
        optimizer.zero_grad()
        if epoch % 5 == 0 :
            print(f'Epoch {epoch + 1:>2} | Train MSE: {loss:.4f}')

        y_test = []
        for snapshot in test_dataset :
            y_hat = snapshot.y.numpy()
            y_hat = utils.inverse_zscore(y_hat, speed.mean(axis = 0), speed.std(axis = 0))
            y_test = np.append(y_test, y_hat)

        model.eval()
        gnn_pred = []
        for snapshot in test_dataset :
            y_hat = model(snapshot.x.unsqueeze(2), snapshot.edge_index, snapshot.edge_weight).squeeze().detach().numpy()
            y_hat = utils.inverse_zscore(y_hat, speed.mean(axis = 0), speed.std(axis = 0))
            gnn_pred = np.append(gnn_pred, y_hat)

        rw_pred = []
        for snapshot in test_dataset :
            y_hat = snapshot.x[:, -1].squeeze().detach().numpy()
            y_hat = utils.inverse_zscore(y_hat, speed.mean(axis=0), speed.std(axis=0))
            rw_pred = np.append(rw_pred, y_hat)

        ha_pred = []
        for i in range(lags, speed_norm.shape[0] - horizon) :
            y_hat = speed_norm.to_numpy()[i - lags:i].T.mean(axis = 1)
            y_hat = utils.inverse_zscore(y_hat, speed.mean(axis = 0), speed.std(axis = 0))
            ha_pred.append(y_hat)

        ha_pred = np.array(ha_pred).flatten()[-len(y_test)]

        print(f'GNN MAE = {utils.MAE(gnn_pred, y_test):.4f}')
        print(f'GNN RMSE = {utils.RMSE(gnn_pred, y_test):.4f}')
        print(f'GNN MAPE = {utils.MAPE(gnn_pred, y_test):.4f}')