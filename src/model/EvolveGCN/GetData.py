class get :
    def __init__(self) :
        from torch_geometric_temporal.signal import temporal_signal_split
        from torch_geometric_temporal.dataset import WikiMathsDatasetLoader

        dataset = WikiMathsDatasetLoader().get_dataset()
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio = 0.5)
        self._get = {'dataset': dataset, 'train': train_dataset, 'test' : test_dataset}

    def __getitem__(self, item):
        return self._get[item]
