import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from train.train_trading import train_trading
from train.train_anomaly import train_anomaly

if __name__ == "__main__":
    train_trading()
    train_anomaly()

