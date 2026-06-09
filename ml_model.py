import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def create_classification_target(df):
    df = df.copy()

    df["Next_Close"] = df["Close"].shift(-1)

    df["Target"] = np.where(
        df["Next_Close"] > df["Close"],
        1,
        0
    )

    df.loc[df["Next_Close"].isna(), "Target"] = np.nan

    return df


def prepare_model_data(df, feature_columns):
    model_df = df.dropna(subset=feature_columns + ["Target"]).copy()
    model_df["Target"] = model_df["Target"].astype(int)

    return model_df


def build_machine_learning_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),

        "SVM RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=42
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42
        )
    }

    return models


def split_time_series_data(model_df, feature_columns):
    x = model_df[feature_columns]
    y = model_df["Target"]

    split_index = int(len(model_df) * 0.8)

    x_train = x.iloc[:split_index]
    x_test = x.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return x_train, x_test, y_train, y_test


def train_and_evaluate_models(model_df, feature_columns):
    x_train, x_test, y_train, y_test = split_time_series_data(
        model_df=model_df,
        feature_columns=feature_columns
    )

    if y_train.nunique() < 2:
        raise ValueError(
            "Data training hanya memiliki satu kelas target. "
            "Gunakan dataset historis yang lebih panjang."
        )

    if y_test.nunique() < 2:
        raise ValueError(
            "Data testing hanya memiliki satu kelas target. "
            "Gunakan dataset historis yang lebih panjang."
        )

    models = build_machine_learning_models()

    evaluation_results = []
    trained_models = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)

        y_prediction = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_prediction)
        precision = precision_score(y_test, y_prediction, zero_division=0)
        recall = recall_score(y_test, y_prediction, zero_division=0)
        f1 = f1_score(y_test, y_prediction, zero_division=0)

        evaluation_results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

        trained_models[model_name] = model

    evaluation_df = pd.DataFrame(evaluation_results)

    comparison_best_row = evaluation_df.sort_values(
        by=["F1-Score", "Accuracy"],
        ascending=False
    ).iloc[0]

    comparison_best_model_name = comparison_best_row["Model"]

    final_model_name = "Random Forest"
    final_model = trained_models[final_model_name]

    return (
        evaluation_df,
        trained_models,
        comparison_best_model_name,
        final_model_name,
        final_model
    )


def normalize_score(value, min_value, max_value):
    if max_value == min_value:
        return 0

    score = (value - min_value) / (max_value - min_value)
    score = max(0, min(1, score))

    return score


def calculate_technical_signal_score(latest_row, historical_df):
    user_open = latest_row["Open"]
    user_high = latest_row["High"]
    user_low = latest_row["Low"]
    user_close = latest_row["Close"]
    user_volume = latest_row["Volume"]

    previous_close = historical_df["Close"].iloc[-1]

    price_range = max(user_high - user_low, 1e-9)
    candle_body_pct = (user_close - user_open) / max(user_open, 1e-9)
    close_position = (user_close - user_low) / price_range

    return_pct = (user_close - previous_close) / max(previous_close, 1e-9)

    recent_volume_median = historical_df["Volume"].tail(30).median()
    volume_ratio = user_volume / max(recent_volume_median, 1e-9)

    body_score = normalize_score(candle_body_pct, -0.08, 0.18)
    close_position_score = normalize_score(close_position, 0.25, 0.95)
    return_score = normalize_score(return_pct, -0.05, 0.18)
    volume_score = normalize_score(volume_ratio, 0.5, 2.5)

    ma5 = latest_row["MA5"]
    ma10 = latest_row["MA10"]
    ma20 = latest_row["MA20"]

    if ma5 > ma10 and ma10 > ma20:
        ma_score = 0.90
    elif ma5 > ma20:
        ma_score = 0.70
    else:
        ma_score = 0.35

    macd = latest_row["MACD"]
    macd_signal = latest_row["MACD_Signal"]

    if macd > macd_signal:
        macd_score = 0.80
    else:
        macd_score = 0.30

    rsi = latest_row["RSI"]

    if rsi < 30:
        rsi_score = 0.65
    elif 30 <= rsi <= 70:
        rsi_score = 0.80
    else:
        rsi_score = 0.55

    bullish_signal = (
        0.24 * body_score +
        0.24 * close_position_score +
        0.16 * return_score +
        0.14 * volume_score +
        0.08 * ma_score +
        0.07 * macd_score +
        0.07 * rsi_score
    ) * 100

    bullish_signal = max(0, min(100, bullish_signal))
    bearish_signal = 100 - bullish_signal

    return bullish_signal, bearish_signal


def calculate_adjusted_confidence(
    prediction,
    bullish_probability,
    bearish_probability,
    bullish_signal,
    bearish_signal
):
    if prediction == 1:
        raw_model_probability = bullish_probability
        technical_confirmation = bullish_signal
    else:
        raw_model_probability = bearish_probability
        technical_confirmation = bearish_signal

    adjusted_confidence = (
        0.40 * raw_model_probability +
        0.60 * technical_confirmation
    )

    adjusted_confidence = max(50, min(95, adjusted_confidence))

    return adjusted_confidence, raw_model_probability, technical_confirmation


def predict_manual_ohlcv_input(
    historical_df,
    user_open,
    user_high,
    user_low,
    user_close,
    user_volume,
    model,
    feature_columns,
    add_technical_indicators
):
    latest_date = historical_df["Date"].max()
    next_date = latest_date + pd.Timedelta(days=1)

    user_row = pd.DataFrame([{
        "Date": next_date,
        "Open": user_open,
        "High": user_high,
        "Low": user_low,
        "Close": user_close,
        "Volume": user_volume,
        "Market Cap": user_close * user_volume
    }])

    combined_df = pd.concat([historical_df, user_row], ignore_index=True)
    combined_df = add_technical_indicators(combined_df)

    prediction_df = combined_df.dropna(subset=feature_columns).copy()

    latest_features = prediction_df[feature_columns].iloc[[-1]]
    latest_row = prediction_df.iloc[-1]

    prediction = model.predict(latest_features)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(latest_features)[0]

        bearish_index = list(model.classes_).index(0)
        bullish_index = list(model.classes_).index(1)

        bearish_probability = probabilities[bearish_index] * 100
        bullish_probability = probabilities[bullish_index] * 100
    else:
        bearish_probability = 0
        bullish_probability = 0

    bullish_signal, bearish_signal = calculate_technical_signal_score(
        latest_row=latest_row,
        historical_df=historical_df
    )

    confidence, raw_model_probability, technical_confirmation = calculate_adjusted_confidence(
        prediction=prediction,
        bullish_probability=bullish_probability,
        bearish_probability=bearish_probability,
        bullish_signal=bullish_signal,
        bearish_signal=bearish_signal
    )

    label = "Bullish" if prediction == 1 else "Bearish"

    return (
        label,
        confidence,
        latest_row,
        bullish_probability,
        bearish_probability,
        bullish_signal,
        bearish_signal,
        raw_model_probability,
        technical_confirmation
    )