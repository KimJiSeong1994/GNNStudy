class get :
    def __init__(self) :
        from src.config import config
        import torch_geometric.transforms as T
        from torch_geometric.datasets import DBLP

        metapahts = [[('author', 'paper'), ('paper', 'author')]]
        transform = T.AddMetaPaths(metapaths = metapahts, drop_orig_edge_types = True)
        dataset = DBLP(root = config.DATAPATH, transform = transform)

        self._get = {'data': dataset, 'data': dataset[0]}

    def __getitem__(self, item) :
        return self._get[item]