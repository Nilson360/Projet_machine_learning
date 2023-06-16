import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Importer le fichier CSV
url = "https://raw.githubusercontent.com/Nilson360/Projet_machine_learning/main/data.csv"
data = pd.read_csv(url)

# Visualiser les premières lignes du DataFrame
print(data.head())

# Vérification des informations sur les données
print(data.info())

# Nettoyage des données : suppression des lignes avec des valeurs manquantes
data = data.dropna()

# Transformation des données qualitatives
# Ici, nous supposons que toutes les colonnes de type object sont des variables qualitatives.
# Nous les transformons en utilisant LabelEncoder

le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

# Sélection des colonnes numériques
numerical_columns = data.select_dtypes(include=np.number).columns
numerical_data = data[numerical_columns]

# Aperçu statistique des données numériques
print(numerical_data.describe())

# Affichage des premières lignes du DataFrame numérique
print(numerical_data.head())

# Tracé de la matrice de corrélation
plt.figure(figsize=(10, 10))
sns.heatmap(numerical_data.corr(), annot=True)
plt.show()

# Histogrammes pour les colonnes numériques
for column in numerical_columns:
    plt.figure()
    data[column].hist()
    plt.title(column)
    plt.show()
