def add_indicators(df):
    df["SMA"] = df.groupby("Asset")["Close"].rolling(10).mean().reset_index(0, drop=True)
    df["Momentum"] = df.groupby("Asset")["Close"].diff(5)
    df = df.fillna(0)
    return df
