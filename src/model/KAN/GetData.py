class get :
    def __init__(self) :
        import torch
        import numpy as np
        import pandas as pd
        from src.model.KAN.args import arg
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.model_selection import train_test_split
        scaler = StandardScaler()

        train_data = pd.read_csv(f'{arg.PATH}/train.csv', sep = ',', encoding = 'utf-8')
        test_data = pd.read_csv(f'{arg.PATH}/test.csv', sep = ',', encoding = 'utf-8')

        label_encoder = LabelEncoder()
        label_encoder_idx = ['Name', 'Sex', 'Ticket']

        train_data[label_encoder_idx] = train_data[label_encoder_idx].apply(label_encoder.fit_transform)
        test_data[label_encoder_idx] = test_data[label_encoder_idx].apply(label_encoder.fit_transform)

        train_data.fillna(train_data.select_dtypes(include = [np.number]).mean(), inplace = True) # ** select_dtypes func. **
        test_data.fillna(test_data.select_dtypes(include = [np.number]).mean(), inplace = True)

        train_df = train_data.drop(arg.DROP_INDEX, axis = 1)
        target_df = train_data[arg.TARGET_INDEX]

        X_train, X_test, y_train, y_test = train_test_split(train_df, target_df, test_size = .2, random_state = 42)
        X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

        to_tensor = lambda x: torch.tensor(x, dtype = torch.float32)
        to_label_tensor = lambda x: torch.tensor(x, dtype = torch.float32).unsqueeze(1)

        self.train_input, self.test_input = to_tensor(X_train), to_tensor(X_test)
        self.train_label, self.test_label = to_label_tensor(y_train.values), to_label_tensor(y_test.values)
        self._get = {'train_input' : self.train_input, 'train_label' : self.train_label, 'test_input' : self.test_input, 'test_label' : self.test_label}

    def __getitem__(self, item) :
        return self._get[item]

    def __len__(self) :
        return len(self.train_input)