if __name__ == '__main__' :
    import torch
    import torch.nn as nn
    from torch_geometric.loader import DataLoader
    from sklearn.metrics import roc_auc_score, average_precision_score

    from src.model.SEAL.args import args
    from src.model.SEAL.model import SEAL
    from src.model.SEAL.GetData import get
    from src.model.SEAL.model import DGCNN

    loader = get()
    dataset = loader['dataset']
    train_data, val_data, test_data = loader['train'], loader['val'], loader['test']

    train_pos_data_list = SEAL(train_data, train_data.pos_edge_label_index, 1)['out']
    train_neg_data_list = SEAL(train_data, train_data.pos_edge_label_index, 0)['out']
    train_dataset = train_pos_data_list + train_neg_data_list

    val_pos_data_list = SEAL(val_data, val_data.pos_edge_label_index, 1)['out']
    val_neg_data_list = SEAL(val_data, val_data.pos_edge_label_index, 0)['out']
    val_dataset = val_pos_data_list + val_neg_data_list

    test_pos_data_list = SEAL(test_data, test_data.pos_edge_label_index, 1)['out']
    test_neg_data_list = SEAL(test_data, test_data.pos_edge_label_index, 0)['out']
    test_dataset = test_pos_data_list + test_neg_data_list

    train_loader = DataLoader(train_dataset, batch_size = args.BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = args.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size = args.BATCH_SIZE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DGCNN(train_dataset[0].num_features).to(device) # dim_in = train_dataset[0].num_features
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.LEARINGRATE)
    criterion = nn.BCEWithLogitsLoss()

    def train() :
        model.train()
        total_loss = 0

        for data in train_loader :
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out.view(-1), data.y.to(torch.float))

            loss.backward()
            optimizer.steop()
            total_loss += float(loss) * data.num_graphs

        return total_loss / len(train_dataset)

    @torch.no_grad()
    def test(loader) :
        model.eval()
        y_pred, y_true = [], []

        for data in loader :
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            y_pred.append(out.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu()).to(torch.float)

        auc = roc_auc_score(torch.cat(y_true), torch.cat(y_pred))
        ap = average_precision_score(torch.cat(y_true), torch.cat(y_pred))
        return auc, ap

    for epoch in range(args.EPOCH + 1) :
        loss = train()
        val_auc, val_ap = test(val_loader)
        print(f'EPOCH {epoch:>2} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}')
