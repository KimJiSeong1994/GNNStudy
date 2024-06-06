class get :
    def __init__(self) :
        from torch_geometric_temporal.signal import temporal_signal_split
        from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader

        dataset = EnglandCovidDatasetLoader().get_dataset(lags = 14)
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio = 0.8)
        self._get = {'data': dataset, 'train': train_dataset, 'test': test_dataset}


    def __getitem__(self, item):
        return self._get[item]
