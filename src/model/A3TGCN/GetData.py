class get :
    def __init__(self) :
        import os
        import pandas as pd
        from io import BytesIO
        from zipfile import ZipFile
        from urllib.request import urlopen
        from src.config import config

        PATH = os.path.join(config.DATAPATH, 'PeMSD7')
        if not os.path.isdir(PATH) :
            url = 'https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/dataset/PeMSD7_Full.zip'
            with urlopen(url) as zurl :
                with ZipFile(BytesIO(zurl.read())) as zfile :
                    zfile.extractall(config.DATAPATH)

        speed = pd.read_csv(f'{PATH}/PeMSD7_V_1026.csv')
        distances = pd.read_csv(f'{PATH}/PeMSD7_W_1026.csv')

