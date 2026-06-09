import pandas as pd
import numpy as np


def validate_dataset(df):
    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_columns = []

    for column in required_columns:
        if column not in df.columns:
            missing_columns.append(column)

    return missing_columns


def clean_numeric_column(series):
    cleaned_series = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace("nan", np.nan)
        .replace("", np.nan)
    )

    return pd.to_numeric(cleaned_series, errors="coerce")


def load_and_clean_dataset(df):
    df = df.copy()

    missing_columns = validate_dataset(df)

    if len(missing_columns) > 0:
        raise ValueError(f"Kolom dataset tidak lengkap: {missing_columns}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

    for column in numeric_columns:
        df[column] = clean_numeric_column(df[column])

    if "Market Cap" in df.columns:
        df["Market Cap"] = clean_numeric_column(df["Market Cap"])
    else:
        df["Market Cap"] = df["Close"] * df["Volume"]

    df = df.dropna(subset=[
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Market Cap"
    ])

    df = df.sort_values("Date")
    df = df.reset_index(drop=True)

    return df


def get_ohlcv_columns():
    return [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Market Cap"
    ]