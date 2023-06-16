import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, roc_curve, auc, confusion_matrix
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

# Histogrammes pour les colonnes numériques
fig, axs = plt.subplots(len(colonnes_numeriques), 1)
for i, column in enumerate(colonnes_numeriques):
    axs[i].hist(data[column])
    axs[i].set_title(column)
plt.tight_layout()
plt.show()

# Cibler les colonnes spécifiques et calculer la corrélation
target_columns = ['Ventes', 'Ouverture', 'Clients', 'PromotionsClassiques']
subset_data = data[target_columns]
correlation_matrix = subset_data.corr()

# Tracer la carte de chaleur
sns.heatmap(correlation_matrix, annot=True)
plt.title("Matrice de corrélation")
plt.show()

# ALGORITHME DE MACHINE LEARNING :

# Application de l'algorithme de l'arbre de décision
X = donnees_numeriques.drop('Ventes', axis=1)
y = donnees_numeriques['Ventes']

# Entraînement de la machine
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Calcul du RMSE
rmse= np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print('Matrice de Confusion:')
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Matrice de confusion")
plt.show()

# Calcul de la courbe ROC
y_pred_proba = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Tracé de la courbe ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='Courbe ROC (aire = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Caractéristiques de fonctionnement du récepteur')
plt.legend(loc="lower right")
plt.show()
