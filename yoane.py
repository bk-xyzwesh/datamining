
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

SEED = 42
os.makedirs("figures_joan", exist_ok=True)

# Chargement des données

abonnes = pd.read_csv("FitZone_abonnes_nettoyes.csv")

print("Dimensions :", abonnes.shape)
print("\nPremières lignes :")
print(abonnes.head())

print("\nValeurs manquantes :")
print(abonnes.isnull().sum())

# Supprimer les doublons éventuels
abonnes = abonnes.drop_duplicates(subset="abonne_id").copy()


# Sélection des variables
variables_candidates = [
    "anciennete_mois",
    "prix_mensuel_eur",
    "distance_domicile_m",
    "visites_par_mois",
    "seances_coaching",
    "reclamations",
    "surface_m2",
    "ouverte_depuis_mois",
    "nb_coachs",
    "formule",
    "creneau_habituel",
    "parking",
    "ville"
]

variables = [
    c for c in variables_candidates
    if c in abonnes.columns
]

X = abonnes[variables].copy()

print("\nVariables utilisées :")
print(variables)

# Séparation des variables numériques et catégorielles
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# Remplacemnt des valeurs infinies
X[num_cols] = X[num_cols].replace(
    [np.inf, -np.inf], np.nan
)

# Remplacement des valeurs manquantes numériques
imputer_num = SimpleImputer(strategy="median")
X[num_cols] = imputer_num.fit_transform(X[num_cols])

# Remplacement des valeurs manquantes catégorielles
for col in cat_cols:
    X[col] = X[col].fillna("Inconnu").astype(str)

# Transformer les variables catégorielles en variables numériques
X_encoded = pd.get_dummies(
    X,
    columns=cat_cols,
    dtype=int
)

# Standardisation des variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

print("\nDimensions après préparation :", X_scaled.shape)

# ETAPE APPLICATION DE ISOLATION FOREST

# On a supposé que 5 % des abonnés sont potentiellement atypiques.


modele = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=SEED,
    n_jobs=-1
)

modele.fit(X_scaled)

predictions = modele.predict(X_scaled)

abonnes["anomalie"] = predictions
abonnes["statut_anomalie"] = abonnes["anomalie"].map({
    1: "Normal",
    -1: "Atypique"
})

abonnes["score_anomalie"] = modele.decision_function(X_scaled)

nb_total = len(abonnes)

nb_anomalies = (
    abonnes["statut_anomalie"] == "Atypique"
).sum()

nb_normaux = nb_total - nb_anomalies

pourcentage = nb_anomalies / nb_total * 100

print("\n====================================")
print("RESULTATS DE LA DETECTION")
print("====================================")

print("Nombre total d'abonnés :", nb_total)
print("Abonnés normaux :", nb_normaux)
print("Abonnés atypiques :", nb_anomalies)

print("Pourcentage d'anomalies :",
      round(pourcentage, 2), "%")


# Extraction


anomalies = abonnes[
    abonnes["statut_anomalie"] == "Atypique"
].copy()

anomalies = anomalies.sort_values(
    "score_anomalie"
)

colonnes_affichage = [
    c for c in [
        "abonne_id",
        "salle",
        "anciennete_mois",
        "formule",
        "prix_mensuel_eur",
        "distance_domicile_m",
        "visites_par_mois",
        "seances_coaching",
        "reclamations",
        "score_anomalie"
    ] if c in anomalies.columns
]

print("\n=== LISTE DES ABONNES ATYPIQUES ===")
print(anomalies[colonnes_affichage].to_string(index=False))

# Exporter les anomalies
anomalies.to_csv(
    "abonnes_atypiques.csv",
    index=False
)


#   Les caractéristiques des anomalies

print("\n=== COMPARAISON DES PROFILS ===")

variables_comparaison = [
    c for c in [
        "anciennete_mois",
        "prix_mensuel_eur",
        "distance_domicile_m",
        "visites_par_mois",
        "seances_coaching",
        "reclamations"
    ] if c in abonnes.columns
]

comparaison = abonnes.groupby(
    "statut_anomalie"
)[variables_comparaison].mean()

print(comparaison.round(2))

comparaison.to_csv(
    "comparaison_profils_anomalies.csv"
)

#  L'impact économique

print("\n=== IMPACT ECONOMIQUE ===")

if "prix_mensuel_eur" in abonnes.columns:

    # Revenu mensuel théorique associé aux abonnés atypiques
    revenu_anomalies = anomalies[
        "prix_mensuel_eur"
    ].sum()

    revenu_normal = abonnes.loc[
        abonnes["statut_anomalie"] == "Normal",
        "prix_mensuel_eur"
    ].sum()

    revenu_total = abonnes[
        "prix_mensuel_eur"
    ].sum()

    print(
        "Revenu mensuel associé aux abonnés atypiques :",
        round(revenu_anomalies, 2), "euros"
    )

    print(
        "Revenu mensuel associé aux abonnés normaux :",
        round(revenu_normal, 2), "euros"
    )

    print(
        "Revenu mensuel total théorique :",
        round(revenu_total, 2), "euros"
    )

    print(
        "Part du revenu associée aux atypiques :",
        round(revenu_anomalies / revenu_total * 100, 2),
        "%"
    )

# Analyse complémentaire des résiliations
cible = next(
    (c for c in ["resilie_reel", "a_resilie"]
     if c in abonnes.columns),
    None
)

if cible is not None:

    print("\nRésiliations parmi les profils atypiques :")

    print(
        pd.crosstab(
            abonnes["statut_anomalie"],
            abonnes[cible],
            normalize="index"
        ).round(3) * 100
    )


# Représentation grafique 

plt.figure(figsize=(7, 5))

sns.countplot(
    data=abonnes,
    x="statut_anomalie"
)

plt.title("Répartition des abonnés normaux et atypiques")
plt.xlabel("Type de comportement")
plt.ylabel("Nombre d'abonnés")

plt.tight_layout()
plt.savefig(
    "figures_joan/repartition_anomalies.png",
    dpi=150
)
plt.show()

# Prix et fréquentations

if all(c in abonnes.columns for c in [
    "prix_mensuel_eur", "visites_par_mois"
]):

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=abonnes,
        x="visites_par_mois",
        y="prix_mensuel_eur",
        hue="statut_anomalie",
        style="statut_anomalie",
        s=90
    )

    plt.title("Prix mensuel et fréquentation des abonnés")
    plt.xlabel("Visites par mois")
    plt.ylabel("Prix mensuel (€)")

    plt.tight_layout()
    plt.savefig(
        "figures_joan/prix_frequentation.png",
        dpi=150
    )
    plt.show()

#GRAPHIQUE 3 - DISTANCE ET FREQUENTATION


if all(c in abonnes.columns for c in [
    "distance_domicile_m", "visites_par_mois"
]):

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=abonnes,
        x="distance_domicile_m",
        y="visites_par_mois",
        hue="statut_anomalie",
        s=90
    )

    plt.title("Distance domicile-salle et fréquentation")
    plt.xlabel("Distance (mètres)")
    plt.ylabel("Visites par mois")

    plt.tight_layout()
    plt.savefig(
        "figures_joan/distance_frequentation.png",
        dpi=150
    )
    plt.show()

#GRAPHIQUE 4 - COMPARAISON DES PROFILS


comparaison.T.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Comparaison des caractéristiques des abonnés")
plt.ylabel("Valeur moyenne")
plt.xlabel("Variables")
plt.xticks(rotation=35)
plt.legend(title="Profil")

plt.tight_layout()
plt.savefig(
    "figures_joan/comparaison_profils.png",
    dpi=150
)
plt.show()

# Score d'anomalie

plt.figure(figsize=(9, 5))

sns.histplot(
    data=abonnes,
    x="score_anomalie",
    hue="statut_anomalie",
    bins=30,
    element="step",
    stat="count"
)

plt.title("Distribution des scores d'anomalie")
plt.xlabel("Score Isolation Forest")
plt.ylabel("Nombre d'abonnés")

plt.tight_layout()
plt.savefig(
    "figures_joan/scores_anomalies.png",
    dpi=150
)
plt.show()

# EXPORTATION DES RESULTATS

abonnes.to_csv(
    "FitZone_resultats_anomalies.csv",
    index=False
)

print("\n====================================")
print("ANALYSE TERMINEE")
print("====================================")

print("Fichiers générés :")
print("- abonnes_atypiques.csv")
print("- comparaison_profils_anomalies.csv")
print("- FitZone_resultats_anomalies.csv")
print("- Dossier figures_joan/")
