import torch
import numpy as np
import pandas as pd

from models.gnn import SimpleGNN
from models.dqn import DQN
from utils.graph import build_adj_matrix
from utils.indicators import add_indicators
from config import ASSETS, NUM_ASSETS, STATE_DIM, ACTION_DIM
from paths import PRICE_CSV

def train_trading():
    df = pd.read_csv(PRICE_CSV)
    df = add_indicators(df)

    # 构建价格矩阵 (T, N)
    price_matrix = df.pivot(index="Date", columns="Asset", values="Close").values

    # 节点特征矩阵 (T, N, STATE_DIM)
    feature_matrix = []
    for asset in ASSETS:
        sub = df[df["Asset"] == asset][["Close", "SMA", "Momentum"]].values
        feature_matrix.append(sub)

    feature_matrix = np.stack(feature_matrix, axis=1)

    adj = build_adj_matrix(price_matrix)

    gnn = SimpleGNN(STATE_DIM, 16)
    dqn = DQN(16, ACTION_DIM)

    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(dqn.parameters()), lr=0.001
    )

    total_reward = 0

    for t in range(20, len(price_matrix) - 1):
        x = torch.tensor(feature_matrix[t], dtype=torch.float)
        emb = gnn(x, adj)

        state = emb.mean(dim=0)
        q_values = dqn(state)
        action = torch.argmax(q_values).item()

        reward = price_matrix[t + 1].mean() - price_matrix[t].mean()
        total_reward += reward * (action - 1)

        loss = -q_values[action] * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[Trading] Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    train_trading()
