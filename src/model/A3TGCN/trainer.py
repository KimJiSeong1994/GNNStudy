if __name__ == '__main__' :
    import numpy as np
    from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split

    from src.model.A3TGCN.GetData import get
    from src.model.A3TGCN.Utils import utils

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

