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