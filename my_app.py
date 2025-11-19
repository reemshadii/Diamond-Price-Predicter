import streamlit as st
import numpy as np
import joblib
import pandas as pd 
import xgboost as xgb

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="💎 Diamond Price Predictor",
    page_icon="💎",
    layout="centered"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_models():
    pipeline = joblib.load("xgboost_pipeline.pkl")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("diamond_price_xbg_model.json")
    return pipeline, xgb_model

# -------------------- TITLE --------------------
st.title("💎 Diamond Price Predictor")
st.write("Enter the diamond’s features below to estimate its price.")

# -------------------- FEATURE INPUTS --------------------
st.subheader("Diamond Features")

col1, col2, col3 = st.columns(3)

# Dropdowns for categorical features
cut = col1.selectbox("Cut", ['Premium', 'Good', 'Very Good', 'Ideal', 'Fair'])
color = col2.selectbox("Color", ['G', 'H', 'F', 'J', 'D', 'I', 'E'])
clarity = col3.selectbox("Clarity", ['VS2', 'VVS2', 'SI2', 'VS1', 'SI1', 'VVS2', 'VVS1', 'IF', 'I1'])

st.markdown("---")

# Numeric inputs with clear buttons
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

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict Price"):
    features = [carat, depth, table, x, y, z]

    if any(v is None for v in features):
        st.error("Please enter valid numeric values for all fields.")
    else:
        # Create the 'xyz' feature
        xyz = x * y * z

        # Create a pandas DataFrame with the correct column names and order
        input_data = pd.DataFrame([[carat, cut, color, clarity, depth, table, xyz]],
                                  columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'xyz'])


        try:
            prediction = model.predict(input_data)[0]
            formatted_price = f"${prediction:,.2f}"
            st.success(f"💰 **Estimated Price:** {formatted_price}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Built using Streamlit and XGBoost.")
