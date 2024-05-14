if __name__ == '__main__' :
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import PowerTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, classification_report, confusion_matrix

    from src.model.anomaly_detection_HetG.GetData import get

    loader = get()
    df = loader['data']
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
    df[['ACK', 'PSH', 'RST', 'SYN',' FIN']] = pd.DataFrame(ohv, columns = ['ACK', 'PSH', 'RST', 'SYN',' FIN'])

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
