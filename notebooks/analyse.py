import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pyplot
print("bibliotheques importees avec succes !")

# 1. Chargement des données

abonnes = pd.read_csv("data/raw/FitZone_abonnes.csv")
salles = pd.read_csv("data/raw/FitZone_salles.csv")

# 2. Afficher les premières lignes

print("Données des abonnés :")
print(abonnes.head())

print("\nDonnées des ventes :")
print(salles.head())

  # Dimensions des données
print("Dimensions des abonnés :", abonnes.shape)
print("Dimensions des salles :", salles.shape)

# Informations générales
print("\nInformations sur les abonnés :")
abonnes.info()

print("\nInformations sur les salles :")
salles.info()


print(abonnes.isnull().sum())
print(salles.isnull().sum())


# Vérification des doublons

print("Doublons abonnés :", abonnes.duplicated().sum())

print("Doublons salles :", salles.duplicated().sum())




# Afficher les lignes en doublon

doublons = abonnes[abonnes.duplicated()]

print("Lignes en doublon :")
print(doublons)


# suppression des  doublons 
abonnes = abonnes.drop_duplicates()


# Afficher le nombre de lignes et de colonnes après suppression
print("Dimensions après suppression :", abonnes.shape)

# Afficher séparément
print("Nombre de lignes :", abonnes.shape[0])
print("Nombre de colonnes :", abonnes.shape[1])




# afficher le nombre d'abonnes par creneau habituel 
print(abonnes["creneau_habituel"].value_counts())

# Calculer le mode du créneau habituel
mode_creneau = abonnes["creneau_habituel"].mode()[0]

# Remplacer les valeurs manquantes par le mode
abonnes["creneau_habituel"] = abonnes["creneau_habituel"].fillna(mode_creneau)

# Vérifier les valeurs manquantes restantes
print(abonnes["creneau_habituel"].isnull().sum())


# Vérifier le nombre de valeurs manquantes
print(abonnes["creneau_habituel"].isnull().sum())

# Vérifier le nombre d'abonnés dans chaque créneau
print(abonnes["creneau_habituel"].value_counts())



# Afficher les statistiques des séances de coaching
print(abonnes["seances_coaching"].describe())

# Calculer la médiane
print("Médiane :", abonnes["seances_coaching"].median())


# Calculer la médiane des séances de coaching
mediane_coaching = abonnes["seances_coaching"].median()

# Remplacer les valeurs manquantes par la médiane
abonnes["seances_coaching"] = abonnes["seances_coaching"].fillna(mediane_coaching)

# Vérifier le nombre de valeurs manquantes restantes
print(abonnes["seances_coaching"].isnull().sum())


# Vérifier les valeurs manquantes dans le fichier abonnés
print("Valeurs manquantes dans abonnés :")
print(abonnes.isnull().sum())

# Vérifier les valeurs manquantes dans le fichier salles
print("Valeurs manquantes dans salles :")
print(salles.isnull().sum())

# Vérifier les doublons dans les deux fichiers
print("Doublons abonnés :", abonnes.duplicated().sum())
print("Doublons salles :", salles.duplicated().sum())

# Afficher les dimensions finales des deux fichiers
print("Dimensions abonnés :", abonnes.shape)
print("Dimensions salles :", salles.shape)




# Afficher les statistiques descriptives des abonnés
print(abonnes.describe())
# Afficher les abonnés ayant une ancienneté négative
print("Ancienneté négative :")
print(abonnes[abonnes["anciennete_mois"] < 0])

# Afficher les abonnés ayant un prix mensuel égal à 0
print("Prix mensuel égal à 0 :")
print(abonnes[abonnes["prix_mensuel_eur"] == 0])



# Afficher toutes les colonnes des abonnés ayant un prix de 0 €
print(
    abonnes.loc[
        abonnes["prix_mensuel_eur"] == 0
    ].to_string(index=False)
)



# Calculer le prix moyen et le prix minimum par formule
print(
    abonnes.groupby("formule")["prix_mensuel_eur"]
    .agg(["mean", "min", "max"])
)

# Afficher les abonnés ayant une ancienneté négative
print(abonnes[abonnes["anciennete_mois"] < 0].to_string(index=False))


# Afficher la répartition des anciennetés
print(abonnes["anciennete_mois"].value_counts().sort_index())


# Remplacer les anciennetés négatives par des valeurs manquantes
abonnes.loc[abonnes["anciennete_mois"] < 0, "anciennete_mois"] = float("nan")

# Vérifier le nombre de valeurs manquantes
print(abonnes["anciennete_mois"].isnull().sum())


# Compter les abonnés ayant un prix mensuel égal à 0
print("Nombre de prix à 0 :", (abonnes["prix_mensuel_eur"] == 0).sum())

# Afficher les prix minimum et maximum par formule
print(abonnes.groupby("formule")["prix_mensuel_eur"].agg(["min", "max"]))



# Remplacer les prix égaux à 0 par des valeurs manquantes
abonnes.loc[abonnes["prix_mensuel_eur"] == 0, "prix_mensuel_eur"] = float("nan")

# Vérifier le nombre de valeurs manquantes
print("Prix manquants :", abonnes["prix_mensuel_eur"].isnull().sum())





# Vérifier les valeurs manquantes après nettoyage
print(abonnes.isnull().sum())

# Vérifier les dimensions du fichier
print("Dimensions :", abonnes.shape)



abonnes.info()


# Afficher les noms des colonnes et leurs types
print(abonnes.dtypes)


# Afficher les statistiques descriptives
print(abonnes.describe())



# Pourcentage de valeurs manquantes par colonne
pourcentage_manquant = (abonnes.isnull().mean() * 100).round(2)

print(pourcentage_manquant)

print("Nombre de valeurs manquantes par colonne :")
print(abonnes.isnull().sum())


# Calculer les médianes
mediane_anciennete = abonnes["anciennete_mois"].median()
mediane_prix = abonnes["prix_mensuel_eur"].median()

print("Médiane de l'ancienneté :", mediane_anciennete)
print("Médiane du prix mensuel :", mediane_prix)


# Remplacer les valeurs manquantes de l'ancienneté par la médiane
abonnes["anciennete_mois"] = abonnes["anciennete_mois"].fillna(mediane_anciennete)

# Remplacer les valeurs manquantes du prix par la médiane
abonnes["prix_mensuel_eur"] = abonnes["prix_mensuel_eur"].fillna(mediane_prix)

# Vérifier les valeurs manquantes restantes
print("Valeurs manquantes après remplacement :")
print(abonnes.isnull().sum())


print("Dimensions finales des abonnés :", abonnes.shape)
print("Nombre de lignes :", abonnes.shape[0])
print("Nombre de colonnes :", abonnes.shape[1])



# Vérification des valeurs manquantes dans salles
print("Valeurs manquantes dans salles :")
print(salles.isnull().sum())

# Vérification des doublons
print("Nombre de doublons dans salles :", salles.duplicated().sum())

# Vérification des dimensions
print("Dimensions de salles :", salles.shape)

# Informations générales
salles.info()

# Statistiques descriptives
print(salles.describe())


print("Surfaces égales à 0 ou négatives :")
print(salles[salles["surface_m2"] <= 0])

print("Anciennetés négatives :")
print(salles[salles["ouverte_depuis_mois"] < 0])

print("Nombre de coachs égal à 0 ou négatif :")
print(salles[salles["nb_coachs"] <= 0])



print("Répartition des valeurs de parking :")
print(salles["parking"].value_counts())


print("Répartition des formules :")
print(abonnes["formule"].value_counts())

print("\nRépartition des créneaux :")
print(abonnes["creneau_habituel"].value_counts())

print("\nRépartition des résiliations :")
print(abonnes["a_resilie"].value_counts())


# Graphique : répartition des formules
abonnes["formule"].value_counts().plot(kind="bar")

pyplot.title("Répartition des abonnés par formule")
pyplot.xlabel("Formule")
pyplot.ylabel("Nombre d'abonnés")

pyplot.show()


# Graphique : répartition des formules
abonnes["formule"].value_counts().plot(kind="bar")

pyplot.title("Répartition des abonnés par formule")
pyplot.xlabel("Formule")
pyplot.ylabel("Nombre d'abonnés")

pyplot.show()


# Graphique : répartition des abonnés par créneau
abonnes["creneau_habituel"].value_counts().plot(kind="bar")

pyplot.title("Répartition des abonnés par créneau habituel")
pyplot.xlabel("Créneau habituel")
pyplot.ylabel("Nombre d'abonnés")

pyplot.show()


# Graphique : répartition des résiliations
abonnes["a_resilie"].value_counts().plot(kind="bar")

pyplot.title("Répartition des abonnés selon la résiliation")
pyplot.xlabel("Résiliation")
pyplot.ylabel("Nombre d'abonnés")

pyplot.show()


# Analyse descriptive des variables numériques
print("\nStatistiques descriptives des abonnés :")
print(abonnes.describe().T)


# Matrice de corrélation des variables numériques
correlations = abonnes.corr(numeric_only=True)

print("\nMatrice de corrélation :")
print(correlations)


# Préparation des données pour la prédiction

# Variable à prédire
y = abonnes["visites_par_mois"]

# Variables utilisées pour effectuer la prédiction
X = abonnes.drop(columns=["visites_par_mois", "abonne_id"])

print("\nDimensions de X :", X.shape)
print("Dimensions de y :", y.shape)


# Variable cible : ce que nous voulons prédire
y = abonnes["visites_par_mois"]

# Variables explicatives : informations utilisées pour prédire
X = abonnes.drop(columns=["visites_par_mois", "abonne_id"])

print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)














# Séparation des variables X et Y

y = abonnes["visites_par_mois"]

X = abonnes.drop(columns=["visites_par_mois", "abonne_id"])

print("Dimensions de X :", X.shape)
print("Dimensions de Y :", y.shape)


# Identifier les variables catégorielles
print("Variables catégorielles :")
print(X.select_dtypes(include=["object"]).columns)



# Transformer les variables catégorielles en nombres
X = pd.get_dummies(X, dtype=int)

print("Dimensions de X après encodage :", X.shape)


from sklearn.model_selection import train_test_split


# Séparation des données : 80 % entraînement et 20 % test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Données d'entraînement :", X_train.shape)
print("Données de test :", X_test.shape)


from sklearn.linear_model import LinearRegression

print("LinearRegression importé avec succès !")



# Création du modèle de régression linéaire
modele = LinearRegression()

print("Modèle de régression linéaire créé avec succès !")


# Entraînement du modèle
modele.fit(X_train, y_train)

print("Entraînement du modèle terminé !")


# Effectuer les prédictions sur les données de test
y_pred = modele.predict(X_test)

print("Premières prédictions :")
print(y_pred[:10])


# Comparer les prédictions aux valeurs réelles
comparaison = pd.DataFrame({
    "Valeurs réelles": y_test.values,
    "Valeurs prédites": y_pred
})

print("\nComparaison entre les valeurs réelles et prédites :")
print(comparaison.head(10))



# Importer la métrique MAE
from sklearn.metrics import mean_absolute_error

# Calculer l'erreur absolue moyenne
mae = mean_absolute_error(y_test, y_pred)

print("Erreur absolue moyenne (MAE) :", mae)


# Importer la métrique MSE
from sklearn.metrics import mean_squared_error

# Calculer le RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Erreur quadratique moyenne (RMSE) :", rmse)

# Importer la métrique R²
from sklearn.metrics import r2_score

# Calculer le coefficient de détermination
r2 = r2_score(y_test, y_pred)

print("Coefficient de détermination (R²) :", r2)


# Comparaison des valeurs réelles et prédites
pyplot.figure(figsize=(8, 5))

pyplot.scatter(y_test, y_pred, label="Prédictions")

# Ajouter la ligne de prédiction parfaite
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

pyplot.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Prédiction parfaite"
)

pyplot.xlabel("Valeurs réelles")
pyplot.ylabel("Valeurs prédites")
pyplot.title("Comparaison des valeurs réelles et prédites")
pyplot.legend()

pyplot.show()

from sklearn.ensemble import RandomForestRegressor


# Création du modèle Random Forest
modele_rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("Modèle Random Forest créé avec succès !")

# Entraînement du modèle Random Forest
modele_rf.fit(X_train, y_train)

print("Entraînement du modèle Random Forest terminé !")


# Effectuer les prédictions avec Random Forest
y_pred_rf = modele_rf.predict(X_test)

print("Premières prédictions Random Forest :")
print(y_pred_rf[:10])


# Évaluation du modèle Random Forest

# Calcul du MAE
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Calcul du RMSE
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Calcul du R²
r2_rf = r2_score(y_test, y_pred_rf)

# Affichage des résultats
print("\nRésultats Random Forest :")
print("MAE :", mae_rf)
print("RMSE :", rmse_rf)
print("R² :", r2_rf)

# Comparaison des deux modèles

comparaison_modeles = pd.DataFrame({
    "Modèle": ["Régression linéaire", "Random Forest"],
    "MAE": [mae, mae_rf],
    "RMSE": [rmse, rmse_rf],
    "R²": [r2, r2_rf]
})

print("\nComparaison des modèles :")
print(comparaison_modeles)



# Identifier l'importance des variables dans Random Forest
importances = pd.DataFrame({
    "Variable": X_train.columns,
    "Importance": modele_rf.feature_importances_
})

importances = importances.sort_values(
    by="Importance",
    ascending=False
)

print("\nImportance des 10 variables les plus influentes :")
print(importances.head(10))


# Graphique des 10 variables les plus importantes
top_10 = importances.head(10)

pyplot.figure(figsize=(10, 6))
pyplot.barh(top_10["Variable"], top_10["Importance"])

pyplot.xlabel("Importance")
pyplot.ylabel("Variables")
pyplot.title("Les 10 variables les plus importantes pour prédire les visites")

pyplot.gca().invert_yaxis()
pyplot.tight_layout()
pyplot.show()