import requests
import streamlit as st

API_URL = "http://localhost:8000"  # change later when deployed

st.set_page_config(page_title="ModbusTrafficXAI", layout="centered")
st.title("ModbusTrafficXAI — Packet Classification + XAI")

st.markdown("### 1) Choose Model")
model_name = st.selectbox("Model", ["cnn", "attention"], index=0)

st.markdown("### 2) Enter 17 features (bytes)")
st.caption("Enter 17 comma-separated numbers. Example: 0,1,2,...,16")

default_text = ",".join(str(i) for i in range(17))
raw = st.text_area("Features", value=default_text, height=80)

def parse_features(text: str):
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    feats = [float(p) for p in parts]
    return feats

col1, col2 = st.columns(2)

with col1:
    do_predict = st.button("Predict")

with col2:
    do_explain = st.button("Explain (LIME + SHAP)")

st.divider()

if do_predict:
    try:
        features = parse_features(raw)
        payload = {"model": model_name, "features": features}
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
        st.subheader("Prediction Result")
        st.json(r.json())
    except Exception as e:
        st.error(f"Error: {e}")

if do_explain:
    try:
        features = parse_features(raw)
        payload = {
            "model": model_name,
            "features": features,
            "lime_top_k": 10,
            "shap_top_k": 10,
            "shap_nsamples": 200
        }
        r = requests.post(f"{API_URL}/explain", json=payload, timeout=120)
        data = r.json()

        st.subheader("Prediction")
        st.write(f"**Model:** {data.get('model')}")
        st.write(f"**Label:** {data.get('label')}")
        st.write(f"**Probability:** {data.get('probability')}")

        st.subheader("LIME Top Features")
        st.json(data.get("lime_top", []))

        st.subheader("SHAP Top Features")
        st.json(data.get("shap_top", []))

    except Exception as e:
        st.error(f"Error: {e}")

