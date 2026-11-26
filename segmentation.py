#Importations des bibliothèques 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


#Chargement des fichiers csv 

abonnes = pd.read_csv("FitZone_abonnes.csv")
salles = pd.read_csv("FitZone_salles.csv")

print("Données des abonnés :")
print(abonnes.head())

print("\nDonnées des salles :")
print(salles.head())

print("\nDimensions des abonnés :", abonnes.shape)

print("\nDimensions des salles :", salles.shape)


#Nettoyage des données

# Supprimer les doublons d'identifiants
abonnes = abonnes.drop_duplicates(
    subset="abonne_id",
    keep="first"
)

# Afficher les valeurs manquantes
print(abonnes.isnull().sum())

# Remplacer les valeurs manquantes numériques
colonnes_numeriques = abonnes.select_dtypes(
    include=np.number
).columns

abonnes[colonnes_numeriques] = abonnes[
    colonnes_numeriques
].fillna(abonnes[colonnes_numeriques].median())

# Remplacer les valeurs manquantes textuelles
colonnes_textuelles = abonnes.select_dtypes(
    include="object"
).columns

for colonne in colonnes_textuelles:
    abonnes[colonne] = abonnes[colonne].fillna(
        abonnes[colonne].mode()[0]
    )

print("Nettoyage terminé !")


# Sélection des variables 

X = abonnes[
    [
        "anciennete_mois",
        "prix_mensuel_eur",
        "distance_domicile_m",
        "visites_par_mois",
        "seances_coaching",
        "reclamations"
    ]
]


# Standarisation des données
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Standardisation terminée !")


# Nombre de clusters : methode du coude

inerties = []

for k in range(1, 8):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inerties.append(kmeans.inertia_)

plt.plot(range(1, 8), inerties, marker="o")

plt.title("Méthode du coude")
plt.xlabel("Nombre de groupes")
plt.ylabel("Inertie")

plt.show()


# Application de l'algorithme KMeans avec 3 clusters 

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

abonnes["cluster"] = kmeans.fit_predict(X_scaled)

print(abonnes["cluster"].value_counts())