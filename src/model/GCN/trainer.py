if __name__ == '__main__' :
    import pandas as pd
    from src.model.GCN import GetData
    from src.model.GCN.args import arg
    from src.model.GCN.model import GCN

    loader = GetData.get()
    data, dataset = loader['data'], loader['dataset']

    data.train_mask = range(18_000)
    data.val_mask = range(18_001, 20_000)
    data.test_mask = range(20_001, len(data.x))

    df_x = pd.DataFrame(data.x.numpy())
    df_x['label'] = pd.DataFrame(data.y)

    model = GCN(
        dim_in = dataset.num_features,
        dim_h = arg.HIDDEN_SIZE,
        dim_out = dataset.num_classes,
    )

    model.fit(
        data = data,
        epochs = arg.EPOCHS
    )

    test_acc = model.test(data)
    print(f'VanillaGNN test accuracy: {test_acc*100:.2f} %')