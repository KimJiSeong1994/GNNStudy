class GetTrain :
    def __init__(self) :
        from src.model.anomaly_detection_HetG.args import arg

        self.BATCH = arg.BATCH_SIZE
        self.features_host = [f'ipsrc_{i}' for i in range(1, 17)] + [f'ipdst_{i}' for i in range(1, 17)]
        self.features_flow = ['daytime', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Duration', 'Packets', 'Bytes', 'ACK', 'PSH', 'RST', 'SYN', 'FIN', 'ICMP ', 'IGMP ', 'TCP  ', 'UDP  ']
        self.labels = ['benign', 'bruteForce', 'dos', 'pingScan', 'portScan']

    def get_loader(self, df, subgraph_size = 1024) :
        import torch
        import numpy as np
        from torch_geometric.data import HeteroData
        from torch_geometric.loader import DataLoader

        data = []
        n_subgraphs = len(df) // subgraph_size
        for i in range(1, n_subgraphs + 1) :
            subgraph = df[(i-1) * subgraph_size : (i * subgraph_size)]
            src_ip = subgraph['Src IP Addr'].to_numpy()
            dst_ip = subgraph['Dst IP Addr'].to_numpy()

            ip_map = {ip: index for index, ip in enumerate(np.unique(np.append(src_ip, dst_ip)))}
            host_to_flow, flow_to_host = self._get_connections(ip_map, src_ip, dst_ip)

            batch = HeteroData()
            batch['host'].x = torch.Tensor(subgraph[self.features_host].to_numpy()).float()
            batch['flow'].x = torch.Tensor(subgraph[self.features_flow].to_numpy()).float()
            batch['flow'].y = torch.Tensor(subgraph[self.labels].to_numpy()).float()
            batch['host', 'flow'].edge_index = host_to_flow
            batch['flow', 'host'].edge_index = flow_to_host
            data.append(batch)

        return DataLoader(data, batch_size = self.BATCH)

    @staticmethod
    def _get_connections(ip_map, src_ip, dst_ip) :
        import torch
        import numpy as np

        src1 = [ip_map[ip] for ip in src_ip]
        src2 = [ip_map[ip] for ip in dst_ip]
        src = np.column_stack((src1, src2)).flatten()
        dst = list(range(len(src_ip)))
        dst = np.column_stack((dst, dst)).flatten()

        return torch.Tensor([src, dst]).int(), torch.Tensor([dst, dst]).int()