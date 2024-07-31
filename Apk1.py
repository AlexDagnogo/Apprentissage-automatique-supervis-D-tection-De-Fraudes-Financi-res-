import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Charger les dictionnaires de mappage, le modèle et le scaler
with open('label_encoders_mapping1.pkl', 'rb') as f:
    mapping_dict = pickle.load(f)
with open('fraud_detection_model1.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler1.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Définir les colonnes catégorielles sans `TransactionStartTime'
categorical_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 
                       'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 
                       'ChannelId']

# Fonction pour encoder les colonnes catégorielles avec gestion des valeurs inconnues
def encode_column(column_data, mapping):
    return column_data.map(mapping).fillna(-1).astype(int)

# Interface Streamlit
st.markdown("<h1 style='text-align: center;'>Application de détection des fraudes</h1>", unsafe_allow_html=True)

# Description de l'application
st.markdown("<h6 style='text-align: center;'>Cette application utilise un modèle de Forêt Aléatoire pour détecter la fraude dans les transactions.</h6>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Faites glisser et déposez le fichier ici", type="csv")
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.success("Fichier chargé avec succès !")
    
    # Bouton pour lancer la prédiction
    if st.button('Prédire les transactions frauduleuses'):
        # Prétraitement des données
        for column in categorical_columns:
            if column in input_data.columns:
                input_data[column] = encode_column(input_data[column], mapping_dict[column])
        
        input_data['TransactionStartTime'] = pd.to_datetime(input_data['TransactionStartTime'])
        input_data['Year'] = input_data['TransactionStartTime'].dt.year
        input_data['Month'] = input_data['TransactionStartTime'].dt.month
        input_data['Day'] = input_data['TransactionStartTime'].dt.day
        input_data['Hour'] = input_data['TransactionStartTime'].dt.hour
        input_data['Minute'] = input_data['TransactionStartTime'].dt.minute
        input_data['Second'] = input_data['TransactionStartTime'].dt.second
        input_data.drop(columns=['TransactionStartTime'], inplace=True)

        input_data_scaled = scaler.transform(input_data)
        predictions = model.predict(input_data_scaled)
        input_data['FraudResult'] = predictions
        input_data['FraudLabel'] = input_data['FraudResult'].map({0: 'Non-Frauduleuse', 1: 'Frauduleuse'})
        
        # Affichage des données prétraitées
        st.subheader("Données après prétraitement :")
        st.dataframe(input_data.drop(columns=['FraudLabel']))
        
        # Division de la page en deux colonnes
        col1, col2 = st.columns(2)

        # Affichage des prédictions
        col1.subheader("Prédictions des transactions")
        col1.dataframe(input_data[['TransactionId', 'FraudLabel']])

        # Visualisation des résultats avec un graphique
        col2.subheader("Visualisation des résultats")
        fraud_counts = input_data['FraudLabel'].value_counts()
        plt.figure(figsize=(5.5, 8))
        sns.barplot(x=fraud_counts.index, y=fraud_counts.values, palette=['green', 'red'])
        plt.xlabel('Type de transaction')
        plt.ylabel('Nombre de transactions')
        plt.title('Répartition des transactions frauduleuses et non-frauduleuses')
        col2.pyplot(plt)
        
        # Message de confirmation que les mappages ont été appliqués
        st.info("Les mappages ont été appliqués avec succès.")