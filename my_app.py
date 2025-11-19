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
    return pipeline, xgb_model

pipeline, model = load_models()

st.title("💎 Diamond Price Predictor")
st.write("Enter the diamond’s features below to estimate its price.")
st.subheader("Diamond Features")
col1, col2, col3 = st.columns(3)
cut = col1.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = col2.selectbox("Color", ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = col3.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
st.markdown("---")

def numeric_input(label, key, min_val=None, max_val=None):
    val = st.text_input(label, key=key, placeholder="Enter a number")
    clear = st.button("✕", key=f"clear_{key}")
    if clear:
        st.session_state[key] = ""
        st.rerun()
    if val.strip() == "":
        return None
    try:
        val = float(val)
        if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
            st.warning(f"⚠️ {label} should be between {min_val} and {max_val}")
            return None
        return val
    except ValueError:
        st.warning(f"⚠️ Please enter a valid number for {label}")
        return None

carat = numeric_input("Carat", "carat", 0.1, 5.0)
depth = numeric_input("Depth", "depth", 50, 70)
table = numeric_input("Table", "table", 50, 70)
x = numeric_input("X (mm)", "x", 3, 20)
y = numeric_input("Y (mm)", "y", 3, 20)
z = numeric_input("Z (mm)", "z", 2, 15)

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
            st.write("Transformed features:", input_transformed)  # Optional debug
            predicted_log_price = model.predict(input_transformed)[0]
            predicted_price = np.exp(predicted_log_price) - 1
            formatted_price = f"${predicted_price:,.2f}"
            st.success(f"💰 **Estimated Price:** {formatted_price}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

st.markdown("---")
st.caption("Built using Streamlit and XGBoost.")
