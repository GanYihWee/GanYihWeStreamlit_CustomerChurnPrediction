import joblib
import streamlit as st

# Load the model in cache
@st.experimental_memo(ttl=60*5,max_entries=20)
def load_model(name):
    return joblib.load(name)

def get_col():
    return ['customerno', 'gender', 'seniorcitizen', 'partner', 'tenure',
       'phoneservice', 'multiplelines', 'contract', 'paymentmethod',
       'monthlycharges']

def load_scaler(name):
    return joblib.load(name)
