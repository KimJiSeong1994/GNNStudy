if __name__ == '__main__' :
    import torch
    from torch import nn
    from torch_geometric.loader import DataLoader
    from torch_geometric.explain import Explainer, GNNExplainer

    from src.model.GIN import GetData
    from src.model.GIN.model import GIN
    from src.model.GIN.args import arg
    from src.model.GIN.metric import metric

    loader = GetData.get()
    dataset = loader['dataset']

    train_dataset = dataset[:int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8): int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]

    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = True)

    def train(model, loader):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.LR)

        model.train()
        for epoch in range(arg.EPOCHS + 1):
            total_loss = 0
            acc = 0

            for data in train_loader :
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)

                loss = criterion(out, data.y)
                total_loss += loss / len(loader)
                acc += metric.accuracy(out.argmax(dim=1), data.y) / len(loader)

                loss.backward()
                optimizer.step()

            val_loss, val_acc = test(model, val_loader)
            if (epoch % 20 == 0):
                print(f'Epoch {epoch:>3} | Train loss: {total_loss:.2f} | Train Acc: {acc * 100:>5.2f}%'
                      f'Val Loss: {val_loss:.2f} | Val Acc: {val_acc * 100:.2f}%')

        return model

    @torch.no_grad()
    def test(model, loader):
        criterion = nn.CrossEntropyLoss()
        model.eval()
        loss = 0
        acc = 0

        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            loss += criterion(out, data.y) / len(loader)
            acc += metric.accuracy(out.argmax(dim=1), data.y) / len(loader)

        return loss, acc

    NUM_CLASS = dataset.num_classes
    model = GIN(
        dim_h = 32,
        node_feature = dataset.num_features,
        num_classes = NUM_CLASS
    )

    model = train(model, train_loader)

    explainer = Explainer(
        model = model,
        algorithm = GNNExplainer(epochs = 200),
        explanation_type = 'model',
        node_mask_type = 'attributes',
        edge_mask_type = 'object',
        model_config = dict(
            mode = 'multiclass_classification',
            task_level = 'node',
            return_type = 'log_probs',
        ),
    )

    data = dataset[-1]
    explanation = explainer(data.x, data.edge_index)
    explanation.visualize_feature_importance('./figure/feature_importance.png', top_k = 10)
    explanation.visualize_graph('./figure/visualize_subgraphs.png')
