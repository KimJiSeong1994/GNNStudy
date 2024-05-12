class get :
    def __init__(self, tidy = True) :
        import os
        import pandas as pd
        from io import BytesIO
        from zipfile import ZipFile
        from urllib.request import urlopen
        from src.config import config

        PATH = os.path.join(config.DATAPATH, 'CIDDS001')
        if not os.path.isdir(PATH) :
            url = 'https://www.hs-coburg.de/fileadmin/hscoburg/WISENT-CIDDS-001.zip'
            with urlopen(url) as zurl :
                with ZipFile(BytesIO(zurl.read())) as zfile :
                    zfile.extractall(config.DATAPATH)

        self.df = pd.read_parquet(os.path.join(PATH, os.listdir(PATH)[1]))
        if tidy == True : self.df = self._preprocess()
        self._get = {'data' : self.df}

    def __getitem__(self, item) :
        return self._get[item]

    def _preprocess(self) :
        self.df['attack_type'] = self.df['attack_type'].replace('---', 'benign')

