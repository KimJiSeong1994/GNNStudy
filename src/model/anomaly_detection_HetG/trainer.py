if __name__ == '__main__' :
    import numpy as np
    import pandas as pd
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