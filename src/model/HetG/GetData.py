class get :
    def __init__(self) :
        from src.config import config
        import torch_geometric.transforms as T
        from torch_geometric.datasets import DBLP

        metapaths = [[('author', 'paper'), ('paper', 'author')]]
        transform = T.AddMetaPaths(metapaths = metapaths, drop_orig_edge_types = True)
        dataset = DBLP(config.DATAPATH, transform = transform)
        data = dataset[0]
        self._get = {'dataset': dataset, 'data': data}

    def __getitem__(self, item) :
        return self._get[item]
