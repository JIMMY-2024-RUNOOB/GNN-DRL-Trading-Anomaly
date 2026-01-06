import torch
import pandas as pd
import numpy as np

from models.gnn import SimpleGNN
from utils.graph import build_adj_matrix
from utils.indicators import add_indicators
from config import ASSETS, STATE_DIM
from paths import PRICE_CSV

def train_anomaly():
    df = pd.read_csv(PRICE_CSV)
    df = add_indicators(df)

    price_matrix = df.pivot(index="Date", columns="Asset", values="Close").values

    feature_matrix = []
    for asset in ASSETS:
        sub = df[df["Asset"] == asset][["Close", "SMA", "Momentum"]].values
        feature_matrix.append(sub)

    feature_matrix = np.stack(feature_matrix, axis=1)

    adj = build_adj_matrix(price_matrix)
    gnn = SimpleGNN(STATE_DIM, 16)

    scores = []

    for t in range(1, len(feature_matrix)):
        x1 = torch.tensor(feature_matrix[t - 1], dtype=torch.float)
        x2 = torch.tensor(feature_matrix[t], dtype=torch.float)

        e1 = gnn(x1, adj)
        e2 = gnn(x2, adj)

        score = torch.norm(e2 - e1).item()
        scores.append(score)

    print(f"[Anomaly] Max score: {max(scores):.2f}")

if __name__ == "__main__":
    train_anomaly()
