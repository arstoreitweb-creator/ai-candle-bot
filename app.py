import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Next Candle Predictor V2",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Next-Candle Predictor — V2")
st.caption("Machine Learning + Walk-Forward Backtesting")

st.warning(
    "⚠️ This is a probabilistic ML experiment, not a guaranteed trading signal."
)


# ============================================================
# SETTINGS
# ============================================================

st.sidebar.header("⚙️ Model Settings")

min_train_size = st.sidebar.slider(
    "Minimum training candles",
    min_value=30,
    max_value=500,
    value=80,
    step=10
)

n_estimators = st.sidebar.slider(
    "Random Forest trees",
    min_value=50,
    max_value=500,
    value=200,
    step=50
)

confidence_threshold = st.sidebar.slider(
    "Trade confidence threshold",
    min_value=0.50,
    max_value=0.90,
    value=0.60,
    step=0.01
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your OHLC candle CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "CSV must contain: open, high, low, close"
    )
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    required_columns = [
        "open",
        "high",
        "low",
        "close"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        st.error(
            "Missing columns: "
            + ", ".join(missing)
        )

        st.stop()

    # Keep only valid numeric OHLC rows

    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    if len(df) < min_train_size + 20:

        st.error(
            f"Not enough candles. "
            f"Please upload at least "
            f"{min_train_size + 20} candles."
        )

        st.stop()


except Exception as e:

    st.error(
        f"Could not read CSV: {e}"
    )

    st.stop()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    x = data.copy()

    # Basic candle features

    x["body"] = (
        x["close"] - x["open"]
    )

    x["range"] = (
        x["high"] - x["low"]
    )

    x["upper_wick"] = (
        x["high"]
        - x[["open", "close"]].max(axis=1)
    )

    x["lower_wick"] = (
        x[["open", "close"]].min(axis=1)
        - x["low"]
    )

    # Avoid division by zero

    safe_range = x["range"].replace(
        0,
        np.nan
    )

    x["body_ratio"] = (
        x["body"].abs()
        / safe_range
    )

    x["upper_wick_ratio"] = (
        x["upper_wick"].abs()
        / safe_range
    )

    x["lower_wick_ratio"] = (
        x["lower_wick"].abs()
        / safe_range
    )

    # Returns / momentum

    x["return_1"] = (
        x["close"].pct_change(1)
    )

    x["return_2"] = (
        x["close"].pct_change(2)
    )

    x["return_3"] = (
        x["close"].pct_change(3)
    )

    x["return_5"] = (
        x["close"].pct_change(5)
    )

    x["return_10"] = (
        x["close"].pct_change(10)
    )

    # Moving averages

    x["sma_5"] = (
        x["close"]
        .rolling(5)
        .mean()
    )

    x["sma_10"] = (
        x["close"]
        .rolling(10)
        .mean()
    )

    x["sma_20"] = (
        x["close"]
        .rolling(20)
        .mean()
    )

    x["price_vs_sma5"] = (
        x["close"] / x["sma_5"] - 1
    )

    x["price_vs_sma10"] = (
        x["close"] / x["sma_10"] - 1
    )

    x["price_vs_sma20"] = (
        x["close"] / x["sma_20"] - 1
    )

    # Volatility

    x["volatility_5"] = (
        x["return_1"]
        .rolling(5)
        .std()
    )

    x["volatility_10"] = (
        x["return_1"]
        .rolling(10)
        .std()
    )

    # Momentum acceleration

    x["momentum_5"] = (
        x["close"]
        - x["close"].shift(5)
    )

    x["momentum_10"] = (
        x["close"]
        - x["close"].shift(10)
    )

    # Candle direction

    x["candle_direction"] = np.sign(
        x["body"]
    )

    # Target:
    # 1 = next candle closes UP
    # 0 = next candle closes DOWN

    x["target"] = (
        x["close"].shift(-1)
        > x["close"]
    ).astype(int)

    return x


feature_df = create_features(df)


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [

    "body",
    "range",
    "upper_wick",
    "lower_wick",

    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",

    "return_1",
    "return_2",
    "return_3",
    "return_5",
    "return_10",

    "price_vs_sma5",
    "price_vs_sma10",
    "price_vs_sma20",

    "volatility_5",
    "volatility_10",

    "momentum_5",
    "momentum_10",

    "candle_direction"
]


# ============================================================
# PREPARE DATA
# ============================================================

model_df = feature_df.dropna(
    subset=FEATURES + ["target"]
).reset_index(drop=True)

if len(model_df) < min_train_size + 10:

    st.error(
        "After feature preparation there are not enough usable candles."
    )

    st.stop()


X = model_df[FEATURES]
y = model_df["target"]


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

st.subheader("🧪 Walk-Forward Backtesting")

predictions = []
actuals = []
probabilities = []
prediction_indices = []


progress = st.progress(0)

total_steps = len(model_df) - min_train_size


for step, i in enumerate(
    range(
        min_train_size,
        len(model_df)
    )
):

    train_X = X.iloc[:i]
    train_y = y.iloc[:i]

    test_X = X.iloc[
        i:i + 1
    ]

    # Train only on past data

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=2
    )

    model.fit(
        train_X,
        train_y
    )

    probability = model.predict_proba(
        test_X
    )[0]

    prediction = model.predict(
        test_X
    )[0]

    # Probability of UP

    if 1 in model.classes_:

        up_index = list(
            model.classes_
        ).index(1)

        up_probability = float(
            probability[up_index]
        )

    else:

        up_probability = 0.0

    predictions.append(
        int(prediction)
    )

    actuals.append(
        int(y.iloc[i])
    )

    probabilities.append(
        up_probability
    )

    prediction_indices.append(i)

    progress.progress(
        min(
            (step + 1) / total_steps,
            1.0
        )
    )


# ============================================================
# BACKTEST RESULTS
# ============================================================

backtest = pd.DataFrame({

    "index": prediction_indices,

    "actual": actuals,

    "prediction": predictions,

    "up_probability": probabilities
})


backtest["down_probability"] = (
    1
    - backtest["up_probability"]
)


# Signal

backtest["signal"] = np.where(

    backtest["up_probability"]
    >= confidence_threshold,

    "UP",

    np.where(

        backtest["down_probability"]
        >= confidence_threshold,

        "DOWN",

        "NO TRADE"
    )
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    backtest["actual"],
    backtest["prediction"]
)

precision = precision_score(
    backtest["actual"],
    backtest["prediction"],
    zero_division=0
)

recall = recall_score(
    backtest["actual"],
    backtest["prediction"],
    zero_division=0
)


# Trade signals only

trades = backtest[
    backtest["signal"]
    != "NO TRADE"
].copy()


if len(trades) > 0:

    trade_accuracy = (
        (
            (
                (trades["signal"] == "UP")
                & (trades["actual"] == 1)
            )
            |
            (
                (trades["signal"] == "DOWN")
                & (trades["actual"] == 0)
            )
        ).mean()
    )

else:

    trade_accuracy = 0.0


# ============================================================
# DISPLAY METRICS
# ============================================================

st.subheader("📊 Backtest Performance")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "ML Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with c2:

    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

with c3:

    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

with c4:

    st.metric(
        "Signal Accuracy",
        f"{trade_accuracy * 100:.2f}%"
    )


st.write(
    f"Total backtest predictions: **{len(backtest)}**"
)

st.write(
    f"Qualified signals: **{len(trades)}**"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("🔢 Confusion Matrix")

cm = confusion_matrix(
    backtest["actual"],
    backtest["prediction"],
    labels=[0, 1]
)

cm_df = pd.DataFrame(

    cm,

    index=[
        "Actual DOWN",
        "Actual UP"
    ],

    columns=[
        "Predicted DOWN",
        "Predicted UP"
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

st.subheader("🤖 Current ML Prediction")


final_model = RandomForestClassifier(

    n_estimators=n_estimators,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1,

    min_samples_leaf=2
)


final_model.fit(
    X,
    y
)


# Latest candle

latest_X = feature_df[
    FEATURES
].iloc[-1:].copy()


if latest_X.isnull().any().any():

    st.error(
        "Latest candle does not contain enough data for prediction."
    )

    st.stop()


latest_probability = final_model.predict_proba(
    latest_X
)[0]


if 1 in final_model.classes_:

    up_index = list(
        final_model.classes_
    ).index(1)

    up_probability = float(
        latest_probability[up_index]
    )

else:

    up_probability = 0.0


down_probability = (
    1 - up_probability
)


# ============================================================
# SIGNAL DECISION
# ============================================================

if up_probability >= confidence_threshold:

    direction = "UP"

    confidence = up_probability

elif down_probability >= confidence_threshold:

    direction = "DOWN"

    confidence = down_probability

else:

    direction = "NO TRADE"

    confidence = max(
        up_probability,
        down_probability
    )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

p1, p2, p3 = st.columns(3)

with p1:

    st.metric(
        "Next Candle Bias",
        direction
    )

with p2:

    st.metric(
        "UP Probability",
        f"{up_probability * 100:.2f}%"
    )

with p3:

    st.metric(
        "DOWN Probability",
        f"{down_probability * 100:.2f}%"
    )


if direction == "UP":

    st.success(
        f"🟢 ML Signal: UP | "
        f"Probability: {confidence * 100:.2f}%"
    )

elif direction == "DOWN":

    st.error(
        f"🔴 ML Signal: DOWN | "
        f"Probability: {confidence * 100:.2f}%"
    )

else:

    st.info(
        f"⚪ ML Signal: NO TRADE | "
        f"Highest probability: {confidence * 100:.2f}%"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("🧠 Feature Importance")

importance_df = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": final_model.feature_importances_

}).sort_values(
    "Importance",
    ascending=False
)


st.dataframe(
    importance_df,
    use_container_width=True
)


# ============================================================
# BACKTEST TABLE
# ============================================================

st.subheader("📈 Recent Backtest Results")

display_backtest = backtest.tail(30).copy()

display_backtest["up_probability"] = (
    display_backtest["up_probability"]
    * 100
)

display_backtest["down_probability"] = (
    display_backtest["down_probability"]
    * 100
)

display_backtest = display_backtest[
    [
        "actual",
        "prediction",
        "up_probability",
        "down_probability",
        "signal"
    ]
]

st.dataframe(
    display_backtest,
    use_container_width=True
)


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

csv_data = backtest.to_csv(
    index=False
).encode("utf-8")


st.download_button(

    label="⬇️ Download Backtest CSV",

    data=csv_data,

    file_name="ml_backtest_results.csv",

    mime="text/csv"
)


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Important: Historical backtest performance does not guarantee "
    "future results. Market conditions can change, and this model "
    "does not guarantee profitable trades."
)
