if __name__ == '__main__' :
    import os
    import torch
    import numpy as np
    import pandas as pd
    from src.config import config

    from src.model.GCN.NodeRegression import GetData
    from src.model.GCN.NodeRegression.args import arg
    from src.model.GCN.NodeRegression.model import GCNRegreesion

    loader = GetData.get()
    data, dataset = loader['data'], loader['data']
    df = pd.read_csv(os.path.join(config.DATAPATH, 'wikipedia/chameleon/musae_chameleon_target.csv'))
    data.y = torch.tensor(np.log10(df['target']))

    model = GCNRegreesion(
        dim_in = dataset.num_features,
        dim_h = arg.HIDDEN_SIZE,
        dim_out = 1,
    )

    model.fit(
        data = data,
        epochs = arg.EPOCHS
    )

    test_loss = model.test(data)
    print(f'GCN test accuracy: {test_loss:.5f} %')