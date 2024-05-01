class get :
    def __init__(self) :
        from src.config import config
        from torch_geometric.datasets import DBLP

        dataset = DBLP(config.DATAPATH, force_reload = True) # Add force_reload -> HetG dataset
        self._get = {'dataset': dataset, 'data': dataset[0]}

    def __getitem__(self, item):
        return self._get[item]

