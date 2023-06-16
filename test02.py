import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importation du fichier CSV
data = pd.read_csv('/Users/nilsonsimao/Desktop/Projet_machine_learning/data.csv')

# Analyse exploratoire des données
print(data.describe())
sns.pairplot(data)
plt.show()

# Préparation des données
data = data.dropna()
X = data[['Ouverture', 'Clients', 'PromotionsClassiques']]
y = data['Ventes']

# Ingénierie des fonctionnalités
data['Ratio_Clients_Ouverture'] = data['Clients'] / data['Ouverture']
X['Ratio_Clients_Ouverture'] = X['Clients'] / X['Ouverture']

# Modélisation et évaluation
models = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor())
]

for name, model in models:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {name}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    print()

    # Affichage de l'histogramme des valeurs prédites par rapport aux valeurs réelles
    plt.figure()
    plt.hist(y_test, alpha=0.5, label='Vraies valeurs')
    plt.hist(y_pred, alpha=0.5, label='Valeurs prédites')
    plt.legend()
    plt.title(f"{name}: Comparaison des valeurs réelles et prédites")
    plt.show()

    # Validation croisée
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-Validation R² scores: {scores}")
    print(f"Cross-Validation R² mean: {scores.mean()}")
    print()
