import os

# 项目根目录（GNN_DRL_Project）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 子目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
ENV_DIR = os.path.join(PROJECT_ROOT, "env")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "train")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# 常用文件路径
PRICE_CSV = os.path.join(DATA_DIR, "prices.csv")
