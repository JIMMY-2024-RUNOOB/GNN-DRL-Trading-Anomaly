class TradingEnv:
    def __init__(self, prices):
        self.prices = prices
        self.t = 0

    def step(self, action):
        price_now = self.prices[self.t]
        price_next = self.prices[self.t + 1]

        reward = (price_next - price_now) * (action - 1)
        self.t += 1

        done = self.t >= len(self.prices) - 2
        return reward, done