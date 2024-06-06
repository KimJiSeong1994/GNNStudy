if __name__ == '__main__' :
    import pandas as pd

    import torch
    from torch_geometric.utils import to_dense_adj

    from src.model.VanillaGNN import GetData
    from src.model.VanillaGNN.args import arg
    from src.model.VanillaGNN.model import MLP

    loader = GetData.get()
    data, dataset = loader['data'], loader['data']

    data.train_mask = range(18_000)
    data.val_mask = range(18_001, 20_000)
    data.test_mask = range(20_001, len(data.x))

    df_x = pd.DataFrame(data.x.numpy())
    df_x['label'] = pd.DataFrame(data.y)
    Ahat = to_dense_adj(data.edge_index)[0]
    Ahat += torch.eye(len(Ahat))

    model = MLP(
        dim_in = dataset.num_features,
        dim_h = arg.HIDDEN_SIZE,
        dim_out = dataset.num_classes,
    )

    model.fit(
        data = data,
        A = Ahat,
        epochs = arg.EPOCHS
    )

    test_acc = model.test(data, Ahat)
    print(f'VanillaGNN test accuracy: {test_acc*100:.2f} %')
