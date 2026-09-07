import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AI Next Candle Predictor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Next-Candle Predictor")
st.write("Experimental market-direction analysis")

st.warning(
    "Prediction is probabilistic and is NOT a guaranteed trading signal."
)

uploaded_file = st.file_uploader(
    "Upload your OHLC candle CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Upload a CSV containing: open, high, low, close"
    )
    st.stop()

try:
    df = pd.read_csv(uploaded_file)

    required = ["open", "high", "low", "close"]

    missing = [
        x for x in required
        if x not in df.columns
    ]

    if missing:
        st.error(
            "Missing columns: " + ", ".join(missing)
        )
        st.stop()

    df = df.copy()

    # Candle structure
    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]

    df["upper_wick"] = (
        df["high"]
        - df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1)
        - df["low"]
    )

    df["body_ratio"] = np.where(
        df["range"] > 0,
        abs(df["body"]) / df["range"],
        0
    )

    # Momentum Pressure Score
    momentum = (
        df["body_ratio"]
        * np.sign(df["body"])
    )

    recent_momentum = momentum.tail(10).mean()

    mps = float(
        np.clip(
            50 + recent_momentum * 50,
            0,
            100
        )
    )

    # Direction
    if mps >= 60:
        direction = "UP"
        confidence = mps

    elif mps <= 40:
        direction = "DOWN"
        confidence = 100 - mps

    else:
        direction = "NO TRADE"
        confidence = 50

    st.subheader("🔮 Current Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Direction", direction)

    with c2:
        st.metric(
            "Model Score",
            f"{confidence:.1f}%"
        )

    with c3:
        st.metric(
            "Momentum Pressure",
            f"{mps:.1f}"
        )

    st.subheader("📊 Latest Candles")

    st.dataframe(
        df.tail(20),
        use_container_width=True
    )

    st.subheader("🧠 Feature Analysis")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.metric(
            "Average Body Ratio",
            f"{df['body_ratio'].tail(10).mean():.3f}"
        )

    with f2:
        st.metric(
            "Average Range",
            f"{df['range'].tail(10).mean():.6f}"
        )

    with f3:
        st.metric(
            "Momentum",
            f"{recent_momentum:.3f}"
        )

    if direction == "UP":
        st.success("🟢 Model bias: UP")

    elif direction == "DOWN":
        st.error("🔴 Model bias: DOWN")

    else:
        st.info("⚪ Model bias: NO TRADE")

except Exception as e:
    st.error(f"Error processing data: {e}")
