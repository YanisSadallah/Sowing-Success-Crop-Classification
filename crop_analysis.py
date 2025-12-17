import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Chargement du dataset
crops = pd.read_csv("soil_measures.csv")

# Exploration rapide
print(crops.head(15))

# Séparation des caractéristiques et de la cible
X = crops.drop(columns="crop")
y = crops["crop"]

# Division entraînement/test (test_size 0.25, random_state 170)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=170)

# Évaluation de chaque nutriment individuellement
feature_performance = {}
for feature in ["N", "P", "K", "ph"]: 
    logreg = LogisticRegression(max_iter=600, multi_class="multinomial")
    logreg.fit(X_train[[feature]], y_train)
    
    # Prédiction sur la même colonne
    y_prediction = logreg.predict(X_test[[feature]])
    
    # Calcul du F1-score pondéré
    fonescore = metrics.f1_score(y_test, y_prediction, average="weighted")
    feature_performance[feature] = fonescore
    print(f"F1-score pour {feature} : {fonescore}")

# Identification automatique de la meilleure caractéristique
best_feature = max(feature_performance, key=feature_performance.get)
best_predictive_feature = {best_feature: feature_performance[best_feature]}

print("\nMeilleure caractéristique prédictive :")
print(best_predictive_feature)
