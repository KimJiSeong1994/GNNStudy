
class get :
    def __init__(self) :
        from src.config import config
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(config.DATAPATH, name = 'Cora')
