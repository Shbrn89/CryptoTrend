def calculate_rsi(close_price, period=14):
    delta = close_price.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    average_gain = gain.rolling(window=period).mean()
    average_loss = loss.rolling(window=period).mean()

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))

    return rsi


def add_technical_indicators(df):
    df = df.copy()

    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()

    df["RSI"] = calculate_rsi(df["Close"], period=14)

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    df["Daily_Return"] = df["Close"].pct_change()
    df["Price_Range"] = df["High"] - df["Low"]
    df["Open_Close_Diff"] = df["Close"] - df["Open"]

    df["Volatility_5"] = df["Daily_Return"].rolling(window=5).std()
    df["Volatility_10"] = df["Daily_Return"].rolling(window=10).std()

    return df


def get_feature_columns():
    return [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Market Cap",
        "MA5",
        "MA10",
        "MA20",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "Daily_Return",
        "Price_Range",
        "Open_Close_Diff",
        "Volatility_5",
        "Volatility_10"
    ]


def get_market_condition(latest_row):
    rsi = latest_row["RSI"]
    macd = latest_row["MACD"]
    macd_signal = latest_row["MACD_Signal"]
    ma5 = latest_row["MA5"]
    ma20 = latest_row["MA20"]

    if rsi >= 70:
        rsi_condition = "Overbought"
    elif rsi <= 30:
        rsi_condition = "Oversold"
    else:
        rsi_condition = "Netral"

    if macd > macd_signal:
        macd_condition = "Momentum Positif"
    else:
        macd_condition = "Momentum Negatif"

    if ma5 > ma20:
        trend_condition = "Short-term Uptrend"
    else:
        trend_condition = "Short-term Downtrend"

    return rsi_condition, macd_condition, trend_condition