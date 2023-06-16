import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_curve, auc, confusion_matrix
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

# EXPLOITATION DES DONNÉES :

# Aperçu statistique des données
print(donnees_numeriques.describe())

# Affichage des premières lignes du DataFrame à nouveau
print(donnees_numeriques.head())

# Histogrammes pour les colonnes numériques
for column in colonnes_numeriques:
    plt.figure()
    data[column].hist()
    plt.title(column)
    plt.show()

# Cibler les colonnes spécifiques et calculer la corrélation
target_columns = ['Ventes', 'Ouverture', 'Clients', 'PromotionsClassiques']
subset_data = data[target_columns]
correlation_matrix = subset_data.corr()

# Tracer la carte de chaleur
sns.heatmap(correlation_matrix, annot=True)
plt.title("Matrice de correlaction")
plt.show()

# ALGO MACHINE LEARNING :

# Application de l'algorithme de régression logistique
X = donnees_numeriques.drop('Ventes', axis=1)
y = donnees_numeriques['Ventes']

# Entraînement de la machine :
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Prédictions et probabilités
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

# Calcul du RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Calcul de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Tracé de la courbe ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
