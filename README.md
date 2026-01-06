# GNN-DRL for Trading and Market Anomaly Detection

This project combines Graph Neural Networks (GNN) and Deep Reinforcement Learning (DRL)
to solve two financial tasks:

1. Multi-asset trading strategy optimization
2. Financial market anomaly detection

## 1. Trading Strategy (Task 1)
- Assets are modeled as nodes in a graph
- Edge weights represent asset correlations
- A DRL agent (DQN) learns buy/sell/hold decisions
- Performance is evaluated by total reward

Example output:
[Trading] Total Reward: 19.46

## 2. Market Anomaly Detection (Task 2)
- GNN learns normal market structure
- Structural deviations produce anomaly scores
- Large scores indicate potential market stress

Example output:
[Anomaly] Max score: 213.62

## How to Run
```bash
pip install -r requirements.txt
python main.py
