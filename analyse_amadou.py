
import pandas as pd

# ==========================================
# ÉTAPE 1 : Chargement et découverte des données
# ==========================================

# Charger les deux fichiers CSV
abonnes = pd.read_csv("FitZone_abonnes.csv")
salles = pd.read_csv("FitZone_salles.csv")

# Afficher les dimensions
print("Dimensions du fichier abonnés :", abonnes.shape)
print("Dimensions du fichier salles :", salles.shape)

# Afficher les 5 premières lignes
print("\n=== LES 5 PREMIERS ABONNÉS ===")
print(abonnes.head())

print("\n=== LES 5 PREMIÈRES SALLES ===")
print(salles.head())


# ==========================================
# ÉTAPE 2 : Vérification de la qualité des données
# ==========================================

# 1. Vérifier les valeurs manquantes
print("\n=== VALEURS MANQUANTES ABONNÉS ===")
print(abonnes.isnull().sum())

print("\n=== VALEURS MANQUANTES SALLES ===")
print(salles.isnull().sum())

# 2. Vérifier les doublons exacts
print("\n=== DOUBLONS ===")
print("Doublons abonnés :", abonnes.duplicated().sum())
print("Doublons salles :", salles.duplicated().sum())

# 3. Vérifier les types de données
print("\n=== TYPES DE DONNÉES ABONNÉS ===")
print(abonnes.dtypes)

print("\n=== TYPES DE DONNÉES SALLES ===")
print(salles.dtypes)


# ==========================================
# ÉTAPE 3 : Examiner les identifiants répétés
# ==========================================

print("\n=== VÉRIFICATION DES IDENTIFIANTS ===")

# Nombre total de lignes
print("Nombre de lignes :", len(abonnes))

# Nombre d'identifiants différents
print("Abonnés uniques :", abonnes["abonne_id"].nunique())

# Nombre d'occurrences répétées
print(
    "Identifiants répétés :",
    abonnes["abonne_id"].duplicated().sum()
)

# Afficher toutes les lignes dont l'ID apparaît plusieurs fois
doublons_id = abonnes[
    abonnes.duplicated(subset=["abonne_id"], keep=False)
]

print("\nExemples de lignes répétées :")
print(doublons_id.sort_values("abonne_id").head(10))


# ==========================================
# ÉTAPE 4 : Vérification avant nettoyage
# ==========================================

print("\n=== AVANT NETTOYAGE ===")

# Compter les doublons exacts
nb_doublons = abonnes.duplicated().sum()

# Compter les répétitions d'identifiants
nb_ids_repetes = abonnes["abonne_id"].duplicated().sum()

print("Nombre de lignes :", len(abonnes))
print("Doublons exacts :", nb_doublons)
print("Identifiants répétés :", nb_ids_repetes)

print(
    "Répétitions d'ID non expliquées par les doublons exacts :",
    nb_ids_repetes - nb_doublons
)


# ==========================================
# ÉTAPE 5 : Suppression des doublons
# ==========================================

print("\n=== NETTOYAGE DES DOUBLONS ===")

# Conserver le nombre de lignes avant le nettoyage
lignes_avant = len(abonnes)

# Vérifier que les identifiants sont renseignés
# et que les répétitions sont des doublons exacts
ids_valides = abonnes["abonne_id"].notna().all()

if ids_valides and nb_doublons == nb_ids_repetes:

    # Supprimer les lignes entièrement identiques
    abonnes = abonnes.drop_duplicates().copy()

    print("Avant nettoyage :", lignes_avant)
    print("Après nettoyage :", len(abonnes))
    print("Lignes supprimées :", lignes_avant - len(abonnes))

else:
    print("ATTENTION : nettoyage non effectué.")
    print("Certains identifiants sont absents ou répétés")
    print("avec des informations différentes.")
    print("Il faut examiner ces lignes avant de continuer.")


# ==========================================
# ÉTAPE 6 : Vérification des valeurs manquantes
# ==========================================

# Vérifier uniquement si le nettoyage des doublons a réussi
if ids_valides and nb_doublons == nb_ids_repetes:

    print("\n=== CONTRÔLE APRÈS NETTOYAGE ===")

    print("Nombre de lignes :", len(abonnes))

    print(
        "Identifiants différents :",
        abonnes["abonne_id"].nunique()
    )

    print(
        "Doublons exacts restants :",
        abonnes.duplicated().sum()
    )

    print(
        "Identifiants répétés restants :",
        abonnes["abonne_id"].duplicated().sum()
    )

    # Compter les valeurs manquantes restantes
    print("\n=== VALEURS MANQUANTES APRÈS NETTOYAGE ===")
    print(abonnes.isnull().sum())

    # Afficher les abonnés avec un créneau manquant
    print("\n=== ABONNÉS AVEC CRÉNEAU MANQUANT ===")

    print(
        abonnes[
            abonnes["creneau_habituel"].isnull()
        ].head(10)
    )

    # Afficher les abonnés avec des séances manquantes
    print("\n=== ABONNÉS AVEC COACHING MANQUANT ===")

    print(
        abonnes[
            abonnes["seances_coaching"].isnull()
        ].head(10)
    )
