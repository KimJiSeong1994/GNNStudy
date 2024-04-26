import torch
import torch.nn as nn

class SEAL :
    def __init__(self, dataset, edge_lablel_index, y) :
        import numpy as np

        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data

        from scipy.sparse.csgraph import shortest_path
        from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix

        data_list = []
        for src, dst in edge_lablel_index.t().tolist() :
            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph([src, dst], 2, dataset.edge_index, relabel_nodes = True)
            src, dst = mapping.tolist()

            mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
            sub_edge_index = sub_edge_index[:, mask1 & mask2]

            src, dst = (dst, src) if src > dst else (src, dst)
            adj = to_scipy_sparse_matrix(sub_edge_index, num_nodes = sub_nodes.size(0)).tocsr()

            idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
            adj_wo_src = adj[idx, :][:, idx]

            idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
            adj_wo_dst = adj[idx, :][:, idx]

            d_src = shortest_path(adj_wo_dst, directed = False, unweighted = True, indices = src)
            d_src = np.insert(d_src, dst, 0, axis = 0)
            d_src = torch.from_numpy(d_src)

            d_dst = shortest_path(adj_wo_src, directed = False, unweighted = True, indices = dst - 1)
            d_dst = np.insert(d_dst, src, 0, axis = 0)
            d_dst = torch.from_numpy(d_dst)

            dist = d_src + d_dst
            z = 1 + torch.min(d_src, d_dst) + dist // 2 * (dist // 2 + dist % 2 - 1)
            z[src], z[dst], z[torch.isnan(z)] = 1., 1., 0.
            z = z.to(torch.long)

            node_labels = F.one_hot(z, num_classes = 200).to(torch.float)
            node_emb = dataset.x[sub_nodes]
            node_x = torch.cat([node_emb, node_labels], dim = 1)

            data = Data(x=node_x, z=z, edge_index=sub_edge_index, y=y)
            data_list.append(data)

            self.data_list = data_list
            self._get = {'out' : self.data_list}

    def __getitem__(self, item) :
        return self._get[item]

    def __len__(self) :
        return len(self.data_list)

class DGCNN(nn.Module) :
    def __init__(self, dim_in, k = 30) :
        super().__init__()
        import torch_geometric.nn as gnn
        from src.model.SEAL.args import args

        self.gcn1 = gnn.GCNConv(dim_in, args.HIDDNE_DIM)
        self.gcn2 = gnn.GCNConv(args.HIDDNE_DIM, args.HIDDNE_DIM)
        self.gcn3 = gnn.GCNConv(args.HIDDNE_DIM, args.HIDDNE_DIM)
        self.gcn4 = gnn.GCNConv(args.HIDDNE_DIM, 1)

        self.global_pool = gnn.aggr.SortAggregation(k = k)
        self.conv1 = nn.Conv1d(1, 16, 97, 97)
        self.conv2 = nn.Conv1d(16, 32, 5, 1)
        self.maxpool = nn.MaxPool1d(2, 2)

        self.linear1 = nn.Linear(352, 128)
        self.dropout  = nn.Dropout(args.DROPOUT_RATIO)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, x, edge_index, batch) :
        h1 = self.gcn1(x, edge_index).tanh()
        h2 = self.gcn2(h1, edge_index).tanh()
        h3 = self.gcn3(h2, edge_index).tanh()
        h4 = self.gcn4(h3, edge_index).tanh()
        h = torch.cat([h1, h2, h3, h4], dim = -1)

        h = self.global_pool(h, batch)
        h = h.view(h.size(0), 1, h.size(-1))

        h = self.conv1(h).relu()
        h = self.maxpool(h)

        h = self.conv2(h).relu()
        h = h.view(h.size(0), -1)

        h = self.linear1(h).relu()
        h = self.dropout(h)
        h = self.linear2(h).sigmoid()
        return h