import streamlit as st
import pandas as pd
import joblib

# Carica i modelli e i scaler
model1 = joblib.load('coliche_model.joblib')
scaler1 = joblib.load('coliche_scaler.joblib')

model2 = joblib.load('rigurgito_model.joblib')
scaler2 = joblib.load('rigurgito_scaler.joblib')

model3 = joblib.load('stipsi_model.joblib')
scaler3 = joblib.load('stipsi_scaler.joblib')

# Crea l'interfaccia utente
st.title('Risk Prediction Score')

# Crea i campi per l'input
PN = st.number_input('Inserisci PN')
EG = st.number_input('Inserisci EG')
Età_Mat = st.number_input('Inserisci Età Mat')
pH = st.number_input('Inserisci pH')

# Bottone per eseguire le previsioni
if st.button('Calcola'):
    # Prepara i dati per la previsione
    features = pd.DataFrame([[PN, EG, Età_Mat, pH]], columns=['PN', 'EG', 'Età_Mat', 'pH'])
    
    # Esegui le previsioni con i tre modelli, usando i rispettivi scaler
    result1 = model1.predict(scaler1.transform(features))
    result2 = model2.predict(scaler2.transform(features))
    result3 = model3.predict(scaler3.transform(features))

    # Mostra i risultati
    st.write('Risultato Modello 1:', result1[0])
    st.write('Risultato Modello 2:', result2[0])
    st.write('Risultato Modello 3:', result3[0])
