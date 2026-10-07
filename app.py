from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "rul_model.joblib"

st.set_page_config(
    page_title="Turbofan Engine Health Prediction",
    page_icon=":rocket:",
    layout="wide"
)

st.title("Turbofan Engine Health Prediction")
if not MODEL_PATH.exists():
    st.error("Model not found. Run python train_model.py first.")
    st.stop()

required_files = [DATA_DIR / "test_FD001.txt",DATA_DIR / "RUL_FD001.txt"]

if not all(path.exists() for path in required_files):
    st.error("Check that both NASA files are in the data folder.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_test_data():
    names = (["engine_id", "cycle"]+ [f"setting_{i}" for i in range(1, 4)]+ [f"sensor_{i}" for i in range(1, 22)])
    test = pd.read_csv(DATA_DIR / "test_FD001.txt",sep=r"\s+",header=None,names=names).dropna(axis=1, how="all")
    actual_rul = pd.read_csv(DATA_DIR / "RUL_FD001.txt",sep=r"\s+",header=None).iloc[:, 0]
    last_rows = (test.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1).sort_values("engine_id").reset_index(drop=True))

    if len(last_rows) != len(actual_rul):
        raise ValueError("Mismatch between test engines and RUL data.")
    last_rows["actual_RUL"] = actual_rul.to_numpy()
    return test, last_rows

try:
    bundle = load_model()
    model = bundle["model"]
    features = bundle["features"]
    test, last_rows = load_test_data()
    last_rows["predicted_RUL"] = np.clip(model.predict(last_rows[features]),0,None)
except Exception as e:
    st.error(f"Could not load model or data: {e}")
    st.stop()

st.sidebar.header("Engine Controls")
engine_ids = sorted(test["engine_id"].unique())
selected_engine = st.sidebar.selectbox("Select Engine ID",engine_ids,index=0)
engine_history = (test[test["engine_id"] == selected_engine].sort_values("cycle"))
engine_result = (last_rows[last_rows["engine_id"] == selected_engine].iloc[0])
prediction = float(engine_result["predicted_RUL"])
actual = float(engine_result["actual_RUL"])
last_cycle = int(engine_result["cycle"])
st.subheader(f"Engine #{selected_engine}")

col1, col2, col3 = st.columns(3)
col1.metric("Estimated Remaining Life",f"{prediction:.1f} cycles")
col2.metric("Last Observed Cycle",f"{last_cycle}")
col3.metric("Absolute Prediction Error",f"{abs(prediction - actual):.1f} cycles")

st.subheader("Sensor History")

sensor_columns = [f"sensor_{i}" for i in range(1, 22)]
selected_sensors = st.multiselect(
    "Choose sensors to visualize",
    sensor_columns,
    default=["sensor_2", "sensor_3", "sensor_4"]
)

if selected_sensors:
    fig, ax = plt.subplots(figsize=(11, 4))
    for sensor in selected_sensors:
        ax.plot(
            engine_history["cycle"],
            engine_history[sensor],
            label=sensor
        )

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Sensor Reading")
    ax.set_title(f"Sensor History for Engine #{selected_engine}")
    ax.legend(ncol=3)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

else:
    st.warning("Select at least one sensor to show its trend.")

st.subheader("Model Evaluation Across Test Engines")
mae = mean_absolute_error(last_rows["actual_RUL"],last_rows["predicted_RUL"])
st.metric("Official Test-Set MAE",f"{mae:.2f} cycles")
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(last_rows["actual_RUL"],last_rows["predicted_RUL"],alpha=0.8)
upper = max(float(last_rows["actual_RUL"].max()),float(last_rows["predicted_RUL"].max()),1)
ax.plot([0, upper],[0, upper],linestyle="--")

ax.set_xlabel("Actual RUL (cycles)")
ax.set_ylabel("Predicted RUL (cycles)")
ax.set_title("Predicted vs Actual RUL")
ax.grid(alpha=0.25)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

with st.expander("View All Test Engine Predictions"):
    display = last_rows[["engine_id","cycle","predicted_RUL","actual_RUL"]].copy()
    display["absolute_error"] = (display["predicted_RUL"] -display["actual_RUL"]).abs()

    st.dataframe(display.round(2),use_container_width=True,hide_index=True)
    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button("Download Prediction Results as CSV",data=csv,file_name="engine_predictions.csv",mime="text/csv")
st.divider()
