import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("full_pipeline.pkl") 

model = load_model()
if 'page' not in st.session_state:
    st.session_state.page = 'home'

page = st.radio("", ["Home", "About"], index=0, horizontal=True)
st.session_state.page = page.lower()

if st.session_state.page == "home":
    st.title("💎 Diamond Price Predictor")
    st.write("Enter the diamond’s features below to estimate its price.")

    col1, col2, col3 = st.columns(3)
    cut = col1.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = col2.selectbox("Color", ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
    clarity = col3.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

    st.markdown("---")

    def numeric_input(label, key, min_value=None, max_value=None):
        val = st.text_input(label, key=key, placeholder="Enter a number")
        if val.strip() == "":
            return None
        try:
            fval = float(val)
            if min_value is not None and fval < min_value:
                st.warning(f"{label} must be ≥ {min_value}")
                return None
            if max_value is not None and fval > max_value:
                st.warning(f"{label} must be ≤ {max_value}")
                return None
            return fval
        except:
            st.warning(f"Enter a valid number for {label}")
            return None

    carat = numeric_input("Carat", "carat", 0.1, 5.0)
    depth = numeric_input("Depth", "depth", 50, 70)
    table = numeric_input("Table", "table", 50, 70)
    x = numeric_input("X (mm)", "x", 3, 20)
    y = numeric_input("Y (mm)", "y", 3, 20)
    z = numeric_input("Z (mm)", "z", 2, 15)

    if st.button("Predict Price"):
        features = [carat, depth, table, x, y, z]
        if any(v is None for v in features):
            st.error("Please enter valid numeric values for all fields.")
        else:
            xyz = x * y * z
            input_data = pd.DataFrame(
                [[carat, cut, color, clarity, depth, table, xyz]],
                columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'xyz']
            )
            try:
                predicted_price = model.predict(input_data)[0]
                formatted_price = f"${predicted_price:,.2f}"
                st.success(f"💰 **Estimated Price:** {formatted_price}")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")

    st.markdown("---")
    st.caption("Built using Streamlit and XGBoost.")

elif st.session_state.page == "about":
    st.title("About This App")
    st.write("""
    This Diamond Price Predictor app uses an XGBoost model to estimate the price of a diamond 
    based on its features such as carat, cut, color, clarity, and dimensions. 

    The app is built with Streamlit for quick and interactive predictions. 

    Features:
    - Numerical input for carat, depth, table, and dimensions (x, y, z)
    - Categorical input for cut, color, and clarity
    - Real-time price prediction using a pre-trained model
    """)
    st.markdown("---")
    st.caption("Built using Streamlit and XGBoost.")
