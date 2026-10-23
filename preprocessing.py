"""
preprocessing.py
================
Cleaning + feature engineering, mirroring the notebook step-for-step.

Expected raw columns: Date, Product Name, Category, Location, Units Sold,
Units Returned, Price, Discount, Revenue.

Run directly to produce data/processed.csv:
    python preprocessing.py
"""

import numpy as np
import pandas as pd

RAW_PATH = "Walmart_supplement.csv"
PROCESSED_PATH = "data/processed.csv"
SEASONAL_PERIOD = 52  # annual cycle on weekly data


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["Product Name", "Date"]).reset_index(drop=True)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["Date", "Product Name"])
    df["Units Sold"] = df["Units Sold"].clip(lower=0)
    df["Units Returned"] = df["Units Returned"].clip(lower=0)
    df["Price"] = df["Price"].clip(lower=0.01)
    df["Discount"] = df["Discount"].clip(lower=0, upper=0.9)
    df["Revenue"] = (df["Units Sold"] * df["Price"] * (1 - df["Discount"])).round(2)
    df["Return_Rate"] = (df["Units Returned"] / df["Units Sold"].replace(0, np.nan)).fillna(0)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["Date"]
    df["Month"] = dt.dt.month
    df["WeekOfYear"] = dt.dt.isocalendar().week.astype(int)
    df["Quarter"] = dt.dt.quarter
    df["Week_sin"] = np.sin(2 * np.pi * df["WeekOfYear"] / 52)
    df["Week_cos"] = np.cos(2 * np.pi * df["WeekOfYear"] / 52)
    df["Holiday_Season"] = df["Month"].isin([11, 12, 1]).astype(int)
    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Product Name", "Date"]).copy()
    g = df.groupby("Product Name")["Units Sold"]

    for lag in [1, 4, 12, 52]:
        df[f"Sales_Lag_{lag}"] = g.shift(lag)

    df["Rolling_Mean_4"] = g.shift(1).rolling(4).mean().reset_index(level=0, drop=True)
    df["Rolling_Std_4"] = g.shift(1).rolling(4).std().reset_index(level=0, drop=True)
    df["Rolling_Mean_12"] = g.shift(1).rolling(12).mean().reset_index(level=0, drop=True)
    df["Rolling_Mean_52"] = g.shift(1).rolling(52).mean().reset_index(level=0, drop=True)

    df["Price_Lag_1"] = df.groupby("Product Name")["Price"].shift(1)
    df["Discount_Lag_1"] = df.groupby("Product Name")["Discount"].shift(1)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Category_Code"] = df["Category"].astype("category").cat.codes
    df["Location_Code"] = df["Location"].astype("category").cat.codes
    df["Product_Code"] = df["Product Name"].astype("category").cat.codes
    return df


def build_features(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = basic_clean(df)
    df = add_calendar_features(df)
    df = add_lag_rolling_features(df)
    df = encode_categoricals(df)
    return df


if __name__ == "__main__":
    feat = build_features()
    feat.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed {feat.shape[0]} rows x {feat.shape[1]} cols -> {PROCESSED_PATH}")
