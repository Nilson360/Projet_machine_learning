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

# PRÉPARATION DES DONNÉES :

# Vérification des informations sur les données
print(data.info())

# Nettoyage des données : suppression des lignes avec des valeurs manquantes
data = data.dropna()

# Exclusion des colonnes non numériques
colonnes_numeriques = data.select_dtypes(include=np.number).columns
donnees_numeriques = data[colonnes_numeriques]

# EXPLOITATION DES DONNÉES :

# Aperçu statistique des données
print(donnees_numeriques.describe())

# Affichage des premières lignes du DataFrame à nouveau
print(donnees_numeriques.head())

# Cibler les colonnes spécifiques et calculer la corrélation
target_columns = ['Ventes', 'Ouverture', 'Clients', 'PromotionsClassiques']
subset_data = data[target_columns]
correlation_matrix = subset_data.corr()

# Histogrammes en subplot pour les colonnes ciblées
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, column in enumerate(target_columns):
    row = i // 2
    col = i % 2
    axes[row, col].hist(data[column])
    axes[row, col].set_title(column)

plt.tight_layout()
plt.show()

# Tracer la carte de chaleur
plt.figure()
sns.heatmap(correlation_matrix, annot=True)
plt.title("Matrice de corrélation")
plt.show()

# ALGORITHME DE MACHINE LEARNING :

# Application de l'algorithme de l'arbre de décision pour la régression
X = donnees_numeriques.drop('Ventes', axis=1)
y = donnees_numeriques['Ventes']

# Entraînement de la machine
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(min_samples_leaf=10)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Calcul du RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

# Affichage de l'histogramme des valeurs prédites par rapport aux valeurs réelles
plt.figure()
plt.hist(y_test, alpha=0.5, label='Vraies valeurs')
plt.hist(y_pred, alpha=0.5, label='Valeurs prédites')
plt.legend()
plt.title("Comparaison des valeurs réelles et prédites")
plt.show()
