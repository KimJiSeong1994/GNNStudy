class get :
    def __init__(self) :
        from src.config import config

        import torch
        from torch_geometric.datasets import DBLP
        from torch_geometric.data import  HeteroData

        dataset = DBLP(config.DATAPATH)
        self._get = {'dataset': dataset, 'data': dataset[0]}

    def __getitem__(self, item):
        return self._get[item]

