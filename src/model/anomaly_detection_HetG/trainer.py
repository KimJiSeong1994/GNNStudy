if __name__ == '__main__' :
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    from src.model.anomaly_detection_HetG.args import arg
    from src.model.anomaly_detection_HetG.GetData import get
    from src.model.anomaly_detection_HetG.GetTrain import GetTrain
    from src.model.anomaly_detection_HetG.model import HeteroGNN

    loader = get()
    df = loader['data']
    df = df.drop(columns=['Src Pt', 'Dst Pt', 'Flows', 'Tos', 'class', 'attackID', 'attackDescription'])
    df['attackType'] = df['attackType'].replace('---', 'benign')
    df['Date first seen'] = pd.to_datetime(df['Date first seen'])

    df['weekday'] = df['Date first seen'].dt.weekday
    df = pd.get_dummies(df, columns = ['weekday'])

    col_idx = np.where(df.columns.str.contains('weekday_0'))[0][0]
    col_list = list(df.columns[:col_idx]) + ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df.columns = col_list

    df['daytime'] = (
            df['Date first seen'].dt.second +
            df['Date first seen'].dt.minute * 60 +
            df['Date first seen'].dt.hour * 60 * 60
    ) / (24 * 60 * 60)

    df = df.reset_index(drop = True)

    one_hot_flags = lambda input : [1 if char1 == char2 else 0 for char1, char2 in zip('APRSF', input[1:])]
    ohv = list(map(one_hot_flags, df['Flags'].to_numpy()))
    df[['ACK', 'PSH', 'RST', 'SYN', 'FIN']] = pd.DataFrame(ohv, columns = ['ACK', 'PSH', 'RST', 'SYN',' FIN'])

    temp = pd.DataFrame()
    temp['SrcIP'] = df['Src IP Addr'].astype(str)
    temp['SrcIP'][~temp['SrcIP'].str.contains('\d{1,3}\.', regex=True)] = '0.0.0.0'
    temp = temp['SrcIP'].str.split('.', expand = True).rename(columns={2: 'ipsrc3', 3: 'ipsrc4'}).astype(int)[['ipsrc3', 'ipsrc4']]
    temp['ipsrc'] = temp['ipsrc3'].apply(lambda x: format(x, "b").zfill(8)) + temp['ipsrc4'].apply(lambda x: format(x, "b").zfill(8))
    df = df.join(temp['ipsrc'].str.split('', expand=True).drop(columns = [0, 17]).rename(columns = dict(enumerate([f'ipsrc_{i}' for i in range(17)]))).astype('int32'))

    temp = pd.DataFrame()
    temp['DstIP'] = df['Dst IP Addr'].astype(str)
    temp['DstIP'][~temp['DstIP'].str.contains('\d{1,3}\.', regex=True)] = '0.0.0.0'
    temp = temp['DstIP'].str.split('.', expand=True).rename(columns={2: 'ipdst3', 3: 'ipdst4'}).astype(int)[['ipdst3', 'ipdst4']]
    temp['ipdst'] = temp['ipdst3'].apply(lambda x: format(x, "b").zfill(8)) + temp['ipdst4'].apply(lambda x: format(x, "b").zfill(8))
    df = df.join(temp['ipdst'].str.split('', expand = True).drop(columns = [0, 17]).rename(columns = dict(enumerate([f'ipdst_{i}' for i in range(17)]))).astype('int32'))

    m_index = df[pd.to_numeric(df['Bytes'], errors='coerce').isnull() == True].index
    df['Bytes'].loc[m_index] = df['Bytes'].loc[m_index].apply(lambda x: 10e6 * float(x.strip().split()[0]))
    df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce', downcast='integer')

    df = pd.get_dummies(df, prefix='', prefix_sep='', columns=['Proto', 'attackType'])
    labels = ['benign', 'bruteForce', 'dos', 'pingScan', 'portScan']

    df_train, df_test = train_test_split(df, random_state=0, test_size=0.2, stratify=df[labels])
    df_val, df_test = train_test_split(df_test, random_state=0, test_size=0.5, stratify=df_test[labels])

    loader = GetTrain()
    train_loader = loader.get_loader(df_train)
    val_loader = loader.get_loader(df_val)
    test_loadr = loader.get_loader(df_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroGNN(dim_h = 64, dim_out = 5, num_layers = 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    @torch.no_grad()
    def test(loader) :
        model.eval()
        y_pred, y_true = [], []
        n_subgraphs, total_loss = 0, 0

        for batch in loader  :
            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            loss = torch.nn.functional.cross_entropy(out, batch['flow'].y.float())

            y_pred.append(out.argmax(dim = 1))
            y_true.appedn(batch['flow'].y.argmax(dim = 1))

            n_subgraphs += arg.BATCH
            total_loss += float(loss) * arg.BATCH_SIZE

        y_pred = torch.cat(y_pred).cpu()
        y_true = torch.cat(y_true).cpu()
        f1score = f1_score(y_true, y_pred, average = 'macro')
        return (total_loss / n_subgraphs), f1score, y_pred, y_true

    model.train()
    for epoch in range(101) :
        n_subgraphs, total_loss = 0, 0

        for batch in train_loader :
            optimizer.zero_grad()
            batch.to(device)

            out = model(batch.x_dict, batch.edge_index_dict)
            loss = torch.nn.functional.cross_entropy(out, batch['flow'].y.float())
            loss.backward()

            optimizer.step()
            n_subgraphs += arg.BATCH_SIZE
            total_loss += float(loss) * arg.BATCH_SIZE

            if epoch % 10 == 0 :
                val_loss, f1score, _, _ = test(val_loader)
                print(f'Epoch : {epoch} | Train Loss : {total_loss/n_subgraphs:.4f} | Val Loss : {val_loss:.4f} | Val F1-score : {f1score * 100:.4f} %')
