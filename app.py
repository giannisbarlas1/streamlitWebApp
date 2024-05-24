import streamlit as st 
import pandas as pd 
import matplotlib as plt
import seaborn as sn
#titlos Web App
st.title('Εξόρυξη και Ανάλυση Δεδομένων')

#elenxos gia upload arxeiwn
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file, engine='openpyxl')
    else:
        st.error("Μη υποστηριζόμενη μορφή αρχείου")
        return None
# Φόρτωση Δεδομένων
uploaded_file = st.file_uploader("Επιλέξτε ένα αρχείο (CSV ή Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Προεπισκόπηση Δεδομένων")
        st.dataframe(df)