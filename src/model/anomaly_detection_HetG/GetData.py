class get :
    def __init__(self) :
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

        df = pd.read_csv(os.path.join(PATH, os.listdir(PATH)[0]))
        self._get = {'data' : df}

    def __getitem__(self, item) :
        return self._get[item]