class metric :
    @staticmethod
    def accuracy(y_pred, y_ture) :
        import torch
        return torch.sum(y_pred == y_ture) / len(y_ture)