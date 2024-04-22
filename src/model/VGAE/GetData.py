class get :
    def __init__(self) :
        import numpy as np
        from src.config import config

        import torch
        import torch_geometric.transforms as T
        from torch_geometric.datasets import Planetoid
        np.random.seed(42)
        torch.manual_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
            T.RandomLinkSplit(
                num_val = 0.05,
                num_test = 0.1,
                is_undirected = True,
                split_labels = True,
                add_negative_train_samples = False
            ),
        ])

        dataset = Planetoid(config.DATAPATH, name = 'Cora', transform = transform)
        train_data, val_data, test_data = dataset[0]

        self._get = {
            'dataset' : dataset,
            'train' : train_data,
            'val' : val_data,
            'test' : test_data,
        }

    def __getitem__(self, item) :
        return self._get[item]

