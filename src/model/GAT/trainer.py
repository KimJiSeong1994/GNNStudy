if __name__ == '__main__' :
    from src.model.GAT import GetData
    from src.model.GAT.args import arg
    from src.model.GAT.model import GAT

    loader = GetData.get()
    data, dataset = loader['data'], loader['data']

    model = GAT(
        dim_in = dataset.num_features,
        dim_h = arg.HIDDEN_SIZE,
        dim_out = dataset.num_classes,
    )

    model.fit(
        data = data,
        epochs = arg.EPOCHS
    )

    test_acc = model.test(data)
    print(f'GCN test accuracy: {test_acc*100:.2f} %')