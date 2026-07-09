import streamlit as st
import joblib
import numpy as np
import pickle
import pandas as pd
from keras.models import load_model

st.markdown(
    """
    <style>
    .stApp {
        background-color: #90EE90;
        color: brown;   /* default text color */
    }

    h1 {
        color: #1f77b4;  /* blue title */
    }

    label {
        color: #1f77b4 !important;
        font-weight: 600;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label span p {
        color: brown !important;
        font-weight: 600;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

oe = pickle.load(open("oe_ann_bank_churn.pkl", "rb"))   
pe = pickle.load(open("powertransformer_ann_bank_churn.pkl", "rb")) 
scaler = pickle.load(open("scaler_ann_bank_churn.pkl", "rb")) 
#model = pickle.load(open("best_model_keras_ann_bank_churn.pkl", "rb"))  #old method from pickle file

model = load_model("best_model_keras_bank_churn_ann.keras")  #new method from h5 file

st.markdown(
    "<h1 style='text-align: center;'>Bank Churn Predictor</h1>",
    unsafe_allow_html=True
)

st.image("cust_churn.jpg", 
             #caption="Churn Classifier", 
              use_container_width=True)

st.write("This app predicts whether a bank customer will churn or not based on their profile information.") 

creditscore = st.number_input("Credit Score", min_value=300, max_value=850, value=600, step=1)  
grography = st.radio("Geography", ("France", "Spain", "Germany"))
gender = st.radio("Gender", ("Male", "Female"))
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
tenure = st.number_input("Tenure (in years)", min_value=0, max_value=10, value=3, step=1)
balance = st.number_input("Balance", min_value=0.0, value=1000.0, step=100.0)
numofproducts = st.number_input("Number of Products", min_value=1, max_value=5, value=1, step=1)  

hascrcard_map = {"Yes": 1, "No": 0}

hascrcard = st.radio("Has Credit Card?", ("Yes", "No"))
hascrcard = hascrcard_map[hascrcard]

isactivemember_map = {"Yes": 1, "No": 0}
isactivemember = st.radio("Is Active Member?", ("Yes", "No"))
isactivemember = isactivemember_map[isactivemember]

estimatedsalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1.0)

if st.button("Classify Customer Churn Possibility"):
    input_data = pd.DataFrame({
        "creditscore": [creditscore],
        "geography": [grography],
        "gender": [gender],
        "age": [age],
        "tenure": [tenure],
        "balance": [balance],
        "numofproducts": [numofproducts],
        "hascrcard": [hascrcard],
        "isactivemember": [isactivemember],
        "estimatedsalary": [estimatedsalary]
    })

    input_data[['geography','gender']]= oe.transform(input_data[['geography','gender']])
    skewed_cols = ['geography', 'age', 'numofproducts', 'hascrcard']
    input_data_transformed = input_data.copy()
    input_data_transformed[skewed_cols] = pe.transform(input_data[skewed_cols])
    feature_order = ['creditscore', 'geography', 'gender', 'age', 'tenure', 'balance', 'numofproducts', 'hascrcard', 'isactivemember', 'estimatedsalary']
    input_data_transformed = input_data_transformed[feature_order]
    input_data_scaled = scaler.transform(input_data_transformed)

    prediction = model.predict(input_data_scaled)
    prediction_class = (prediction > 0.5).astype(int)

    if prediction_class[0] == 1:
        Prediction_text = "High possibility of Churn[1]."
    else:
        Prediction_text = "Low possibility of Churn[0]."

    if Prediction_text == "High possibility of Churn[1].":
        bg_color = "#ffebee"      # Light red
        border_color = "#d32f2f"  # Red
        text_color = "#b71c1c"    # Dark red
    else:  # Low possibility of Churn
        bg_color = "#e8f0ff"      # Light blue
        border_color = "#0B3D91"  # Blue
        text_color = "#0B3D91"    # Dark blue

    st.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            padding:15px;
            border-radius:10px;
            border:2px solid {border_color};
            color:{text_color};
            font-size:18px;
            font-weight:bold;
            text-align:center;
        ">
            Predicted Customer Churn Possibility: {Prediction_text}
        </div>
        """,
        unsafe_allow_html=True
    )