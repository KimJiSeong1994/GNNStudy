class get :
    def __init__(self) :
        from src.config import config
        import torch_geometric.transforms as T
        from torch_geometric.datasets import WikipediaNetwork
        dataset = WikipediaNetwork(
            config.DATAPATH,
            name = 'chameleon',
            transform = T.RandomNodeSplit(num_val=200, num_test=500)
        )
        data = dataset[0]

        # logging Graph DataSet information
        print(f'Dataset: {dataset}')
        print(f'--------------')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of nods: {data.x.shape[0]}')
        print(f'Number of feature: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')

        print(f'Graph: ')
        print(f'-----')
        print(f'Edges are directed: {data.is_directed()}')
        print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Graph bas loops: {data.has_self_loops()}')

        self._get = {'dataset' : dataset, 'data' : data}

    def __getitem__(self, item) :
        return self._get[item]

