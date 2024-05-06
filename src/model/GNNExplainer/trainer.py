if __name__ == '__main__' :
    import numpy as np

    import torch
    import torch.nn.functional as F
    import torch_geometric.nn as gnn

    from captum.attr import IntegratedGradients
    from torch_geometric.explain import Explainer, GNNExplainer

    from src.model.GNNExplainer.model import GCN
    from src.model.GNNExplainer.GetData import get

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = get()
    dataset, data = loader['dataset'], loader['data']
    model = GCN(
        num_features = dataset.num_features,
        num_classes = dataset.num_classes,
        dim_h = 64
    ).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 5e-4)

    for epoch in range(200) :
        model.train()
        optimizer.zero_grad()
        log_logits = model(data.x, data.edge_index)

        loss = F.nll_loss(log_logits, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 :
            print(f'Epoch :{epoch:>2}, Train Loss : {loss:.4f}')

    def accuracy(pred_y, y) :
        return ((pred_y == y).sum() / len(y)).item()

    @torch.no_grad()
    def test(model, data) :
        model.eval()
        out = model(data.x, data.edge_index)
        acc = accuracy(out.argmax(dim = 1), data.y)
        return acc

    acc = test(model, data)
    print(f'Test Accuracy :{acc * 100:.3f} %')

    node_idx = 0
    captum_model = gnn.to_captum_model(
        model,
        mask_type = 'node_and_edge',
        output_idx = node_idx
    )

    ig = IntegratedGradients(captum_model)
    edge_mask = torch.ones(data.num_edges, requires_grad = True, device = device)

    attr_node, attr_edge = ig.attribute(
        (data.x.unsqueeze(0), edge_mask.unsqueeze(0)), # .unsqueeze func : [n, n] -> [1, n, n]
        target = int(data.y[node_idx]),
        additional_forward_args = (data.edge_index),
        internal_batch_size = 1
    )

    attr_node = attr_node.squeeze(0).abs().sum(dim = 1)
    attr_node /= attr_node.max()

    attr_edge = attr_edge.squeeze(0).abs()
    attr_edge /= attr_edge.max()

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

    explanation = explainer(data.x, data.edge_index)
    explanation.visualize_graph('./figure/twitter_graphs.png')
    explanation.visualize_feature_importance('./figure/twitter_feature_importance.png', top_k = 10)