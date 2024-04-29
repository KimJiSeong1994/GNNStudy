if __name__ == '__main__' :
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    data['user'].x = torch.Tensor(
        [[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3]]
    )

    data['game'].x = torch.Tensor(
        [[1, 1],
         [2, 2]]
    )

    data['dev'].x = torch.Tensor(
        [[1],
         [2]]
    )

    data['user', 'follows', 'user'].edge_index = torch.Tensor([[0, 1], [1, 2]])
    data['user', 'plays', 'game'].edge_index = torch.Tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    data['dev', 'develops', 'game'].edge_index = torch.Tensor([[0, 1], [0, 1]])

    data['user', 'plays', 'game'].edge_attr = torch.Tensor([[2], [0.5], [10], [12]])
