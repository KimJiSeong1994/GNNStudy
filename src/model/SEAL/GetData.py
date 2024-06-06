class get :
    def __init__(self) :
        from src.config import config
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import RandomLinkSplit

        transform = RandomLinkSplit(
            num_val = .05,
            num_test = .1,
            is_undirected = True,
            split_labels = True
        )

        dataset = Planetoid(config.DATAPATH, name = 'Cora', transform = transform)
        train_data, val_data, test_data = dataset[0]

        self._get = {
            'data': dataset,
            'train': train_data,
            'val': val_data,
            'test': test_data,
        }

    def __getitem__(self, item):
        return self._get[item]