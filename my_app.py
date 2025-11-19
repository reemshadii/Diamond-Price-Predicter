import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

st.set_page_config(
    page_title="💎 Diamond Price Predictor",
    page_icon="💎",
    layout="centered"
)

@st.cache_resource
def load_models():
    pipeline = joblib.load("full_pipeline.pkl")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("model.json")
    # Load target scaler if you used one
    try:
        y_scaler = joblib.load("target_scaler.pkl")
    except:
        y_scaler = None
    return pipeline, xgb_model, y_scaler

pipeline, model, y_scaler = load_models()

st.title("💎 Diamond Price Predictor")
st.write("Enter the diamond’s features below to estimate its price.")
st.subheader("Diamond Features")
col1, col2, col3 = st.columns(3)
cut = col1.selectbox("Cut", ['Premium', 'Good', 'Very Good', 'Ideal', 'Fair'])
color = col2.selectbox("Color", ['G', 'H', 'F', 'J', 'D', 'I', 'E'])
clarity = col3.selectbox("Clarity", ['VS2', 'VVS2', 'SI2', 'VS1', 'SI1', 'VVS1', 'IF', 'I1'])
st.markdown("---")

def numeric_input(label, key):
    val = st.text_input(label, key=key, placeholder="Enter a number")
    clear = st.button("✕", key=f"clear_{key}")
    if clear:
        st.session_state[key] = ""
        st.rerun()
    if val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        st.warning(f"⚠️ Please enter a valid number for {label}")
        return None

carat = numeric_input("Carat", "carat")
depth = numeric_input("Depth", "depth")
table = numeric_input("Table", "table")
x = numeric_input("X", "x")
y = numeric_input("Y", "y")
z = numeric_input("Z", "z")

if st.button("🔮 Predict Price"):
    features = [carat, depth, table, x, y, z]
    if any(v is None for v in features):
        st.error("Please enter valid numeric values for all fields.")
    else:
        xyz = x * y * z
        input_data = pd.DataFrame([[carat, cut, color, clarity, depth, table, xyz]],
                                  columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'xyz'])
        try:
            input_transformed = pipeline.transform(input_data)
            predicted_log_price = model.predict(input_transformed)[0]
            predicted_original_price = np.exp(predicted_log_price) - 1
            formatted_price = f"${predicted_original_price:,.2f}"
            st.success(f"💰 **Estimated Price:** {formatted_price}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
st.write("Transformed features:", input_transformed)


st.markdown("---")
st.caption("Built using Streamlit and XGBoost.")
