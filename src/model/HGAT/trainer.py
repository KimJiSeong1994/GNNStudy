if __name__ == '__main__' :
    import torch
    import torch_geometric.nn as gnn

    from src.model.HGAT.model import GAT
    from src.model.HGAT.GetData import get
    from src.model.HGAT.args import arg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = get()
    dataset, data = loader['dataset'], loader['data']

    model = GAT(dim_h = arg.HIDDEN_SIZE, dim_out = arg.OUTPUT_DIM)
    model = gnn.to_hetero(model, data.metadata(), aggr = 'sum')

    data, model = data.to(device), model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = arg.LEARINGRATE, weight_decay = arg.WEIGHT_DECAY)

    @torch.no_grad()
    def test(mask) :
        model.eval()
        pred = model(data.x_dict, data.edge_index_dict)
        acc = (pred[mask] == data['author'].y[mask.sum() / mask.sum())
        return float(acc)



