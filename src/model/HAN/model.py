from torch import nn
class HAN(nn.Module) :
    def __init__(self, dim_in, dim_out, metapath, dim_h = 128, heads = 8) :
        super().__init__()
        import torch_geometric.nn as gnn
        from src.model.HAN.args import arg

        self.han = gnn.HANConv(dim_in, dim_h, heads = heads, dropout = arg.DROPOUT, metadata = metapath)
        self.linear = gnn.Linear(dim_h, dim_out)

    def forward(self, x_dict, edge_index_dict) :
        out = self.han(x_dict, edge_index_dict)
        out = self.linear(out['author'])
        return out

