import numpy as np
import torch

def build_adj_matrix(price_matrix):
    corr = np.corrcoef(price_matrix.T)
    adj = torch.tensor(corr, dtype=torch.float)
    return adj