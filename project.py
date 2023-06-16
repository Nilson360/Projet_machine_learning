import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importer le fichier Excel
data = pd.read_csv('/Users/nilsonsimao/Desktop/Projet_machine_learning/data.csv')

# Afficher les premières lignes du DataFrame

print(data.head())

# PRÉPARATION DES DONNÉES :

# Vérification des informations sur les données
print(data.info())

# Nettoyage des données : suppression des lignes avec des valeurs manquantes:

data = data.dropna()
# Exclusion des colonnes non numériques
colonnes_numeriques = data.select_dtypes(include=np.number).columns
donnees_numeriques = data[colonnes_numeriques]

# EXPLOTATION DE LA DONNÉES :



# Aperçu statistique des données
print(donnees_numeriques.describe())

# Affichage des premières lignes du DataFrame à nouveau
print(donnees_numeriques.head())

# Cibler les colonnes spécifiques et calculer la corrélation
target_columns = ['Ventes','Ouverture','Clients','PromotionsClassiques']
subset_data = data[target_columns]
correlation_matrix = subset_data.corr()

# Tracer la carte de chaleur
sns.heatmap(correlation_matrix, annot=True)
#plt.show()

#sns.heatmap(donnees_numeriques.corr(), annot=True)
plt.show()