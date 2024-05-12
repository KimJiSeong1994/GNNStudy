if __name__ == '__main__' :
    from src.model.anomaly_detection_HetG.GetData import get

    loader = get()
    dataset = loader['data']