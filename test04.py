import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Importation du fichier CSV
data = pd.read_csv('/Users/nilsonsimao/Desktop/Projet_machine_learning/data.csv')

# Affichage des premières lignes du DataFrame
print(data.head())

# Vérification des informations sur les données
print(data.info())

# Nettoyage des données : suppression des lignes avec des valeurs manquantes
data = data.dropna()
print('======================================= Données traités ===========================================')
print(data)
print('==================================================================================')

# Cibler toutes les colonnes numériques
donnees_numeriques = data.select_dtypes(include=np.number)

# Calculer la matrice de corrélation
correlation_matrix = donnees_numeriques.corr()

# Tracer la carte de chaleur
plt.figure()
sns.heatmap(correlation_matrix, annot=True)
plt.title("Matrice de corrélation")
plt.show()