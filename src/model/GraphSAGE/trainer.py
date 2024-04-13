if __name__ == '__main__' :
    from src.model.GraphSAGE import GetData
    from src.model.GraphSAGE.args import arg

    from src.model.GraphSAGE.model import GraphSAGE
    from torch_geometric.loader import NeighborLoader

    loader = GetData.get()
    data, dataset = loader['data'], loader['dataset']

    train_loader = NeighborLoader(
        data,
        num_neighbors = [arg.NUM_NEIGHTBORS, arg.NUM_NEIGHTBORS],
        batch_size = arg.BATCH ,
        input_nodes = data.train_mask
    )

    model = GraphSAGE(
        dataset.num_features,
        arg.HIDDEN_SIZE,
        dataset.num_classes
    )

    model.fit(train_loader, arg.EPOCHS)
    acc = model.test(data)
    print(f'GraphSAGE test accuracy: {acc*100:.2f} %')
