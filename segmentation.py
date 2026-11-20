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

