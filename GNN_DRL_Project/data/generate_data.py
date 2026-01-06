import time
import yfinance as yf
import pandas as pd
import numpy as np
from config import ASSETS, START_DATE, END_DATE
from paths import DATA_DIR, PRICE_CSV
import os


def safe_download():
    all_data = []

    for asset in ASSETS:
        print(f"Downloading {asset} ...")
        df = yf.download(
            asset,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            raise RuntimeError(f"Download failed for {asset}")

        df = df[["Close", "Volume"]]
        df["Asset"] = asset
        df.reset_index(inplace=True)

        all_data.append(df)
        time.sleep(2)

    return pd.concat(all_data, ignore_index=True)


def generate_fallback_data():
    print("Yahoo unavailable, generating fallback data...")
    dates = pd.date_range(START_DATE, END_DATE)
    all_data = []

    for asset in ASSETS:
        price = 100 + np.cumsum(np.random.randn(len(dates)))
        volume = np.random.randint(1000, 5000, len(dates))

        df = pd.DataFrame({
            "Date": dates,
            "Close": price,
            "Volume": volume,
            "Asset": asset
        })
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        df = safe_download()
        print("Yahoo Finance data downloaded.")
    except Exception as e:
        print(f"Yahoo error: {e}")
        df = generate_fallback_data()

    df.to_csv(PRICE_CSV, index=False)
    print(f"Data saved to {PRICE_CSV}")
