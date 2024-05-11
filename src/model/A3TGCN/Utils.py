class utils :
    @staticmethod
    def compute_adj(dist, sigma_2 = .1, epsilon = .5) :
        import numpy as np
        d = dist.to_numpy() / 10000.
        d2 = d * d
        n = dist.shape[0]
        w_mask = np.ones([n, n]) - np.identity(n)
        return np.exp(-d2 / sigma_2) * (np.exp(-d2 / sigma_2) >= epsilon) * w_mask

    @staticmethod
    def zscore(x, mean, std) :
        return (x - mean) / std

    @staticmethod
    def inverse_zscore(x, mean, std) :
        return x * std + mean

    @staticmethod
    def MAE(real, pred) :
        import numpy as np
        return np.mean(np.abs(pred - real))

    @staticmethod
    def RMSE(real, pred) :
        import numpy as np
        return np.sqrt(np.mean(np.abs(pred - real) ** 2))

    @staticmethod
    def MAPE(real, pred) :
        import numpy as np
        return np.mean(np.abs(pred - real) / (real + 1e-5))

