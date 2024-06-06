class get :
    def __init__(self) :
        from src.config import config
        from torch_geometric.datasets import Twitch

        dataset = Twitch(config.DATAPATH, name = 'EN')
        self._get = {'data': dataset, 'data': dataset[0]}

    def __getitem__(self, item):
        return self._get[item]

