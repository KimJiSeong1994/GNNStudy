class get :
    def __init__(self) :
        from src.config import config
        from torch_geometric.datasets import PPI

        train_dataset = PPI(config.DATAPATH, split = 'train')
        val_dataset = PPI(config.DATAPATH, split = 'val')
        test_dataset = PPI(config.DATAPATH, split = 'test')

        self._get = {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}

    def __getitem__(self, item) :
        return self._get[item]

