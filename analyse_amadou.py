
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



# ÉTAPE 6 : Vérification des valeurs manquantes

print("\n=== VALEURS MANQUANTES APRÈS NETTOYAGE ===")
print(abonnes.isnull().sum())

# Afficher les abonnés sans créneau habituel
print("\n=== CRÉNEAUX MANQUANTS ===")
print(
    abonnes[abonnes["creneau_habituel"].isnull()].head(10)
)

# Afficher les abonnés sans nombre de séances de coaching
print("\n=== COACHING MANQUANT ===")
print(
    abonnes[abonnes["seances_coaching"].isnull()].head(10)
)

# ==========================================
# ÉTAPE 7 : Traitement des valeurs manquantes
# ==========================================

print("\n=== TRAITEMENT DES CRÉNEAUX MANQUANTS ===")

# Compter les valeurs manquantes avant traitement
print(
    "Avant traitement :",
    abonnes["creneau_habituel"].isnull().sum()
)

# Remplacer les valeurs manquantes par "Inconnu"
abonnes["creneau_habituel"] = (
    abonnes["creneau_habituel"].fillna("Inconnu")
)

# Vérifier les valeurs manquantes après traitement
print(
    "Après traitement :",
    abonnes["creneau_habituel"].isnull().sum()
)

# ÉTAPE 7 : Traitement des séances de coaching

print("\n=== TRAITEMENT DU COACHING ===")

# Vérifier le nombre de valeurs manquantes
print("Avant traitement :",
      abonnes["seances_coaching"].isnull().sum())

# Calculer la médiane
mediane_coaching = abonnes["seances_coaching"].median()

print("Médiane :", mediane_coaching)

# Remplacer les valeurs manquantes par la médiane
abonnes["seances_coaching"] = (
    abonnes["seances_coaching"].fillna(mediane_coaching)
)

# Vérifier après traitement
print("Après traitement :",
      abonnes["seances_coaching"].isnull().sum())

# ==========================================
# ÉTAPE 8 : Vérification des incohérences
# ==========================================

print("\n=== VÉRIFICATION DES VALEURS NUMÉRIQUES ===")

# Afficher les statistiques des colonnes numériques
print(abonnes.describe())

print("\n=== VÉRIFICATION DES VALEURS NÉGATIVES ===")

# Vérifier les prix négatifs
print("Prix négatifs :",
      (abonnes["prix_mensuel_eur"] < 0).sum())

# Vérifier les distances négatives
print("Distances négatives :",
      (abonnes["distance_domicile_m"] < 0).sum())

# Vérifier les visites négatives
print("Visites négatives :",
      (abonnes["visites_par_mois"] < 0).sum())

# Vérifier les séances de coaching négatives
print("Séances négatives :",
      (abonnes["seances_coaching"] < 0).sum())

# Vérifier les anciennetés négatives
print("Anciennetés négatives :",
      (abonnes["anciennete_mois"] < 0).sum())

# ÉTAPE 8 : Identifier les anciennetés négatives

print("\n=== ABONNÉS AVEC UNE ANCIENNETÉ NÉGATIVE ===")

abonnes_incoherents = abonnes.loc[
    abonnes["anciennete_mois"] < 0,
    ["abonne_id", "anciennete_mois"]
]

print(abonnes_incoherents)


# ==========================================
# ÉTAPE 8 : Correction des anciennetés négatives
# ==========================================

print("\n=== CORRECTION DES ANCIENNETÉS ===")

# Remplacer les anciennetés négatives par NaN
abonnes.loc[
    abonnes["anciennete_mois"] < 0,
    "anciennete_mois"
] = float("nan")

# Calculer la médiane des anciennetés valides
mediane_anciennete = abonnes["anciennete_mois"].median()

print("Médiane :", mediane_anciennete)

# Remplacer les NaN par la médiane
abonnes["anciennete_mois"] = (
    abonnes["anciennete_mois"].fillna(mediane_anciennete)
)

# Vérifier que les erreurs ont disparu
print("Anciennetés négatives restantes :",
      (abonnes["anciennete_mois"] < 0).sum())

print("Valeurs manquantes restantes :",
      abonnes["anciennete_mois"].isnull().sum())

# ==========================================
# ÉTAPE 8 : Vérification des incohérences des salles
# ==========================================

print("\n=== VÉRIFICATION DES SALLES ===")

# Afficher les statistiques des salles
print(salles.describe())

# Vérifier les valeurs incorrectes
print("\n=== INCOHÉRENCES DES SALLES ===")

print("Surfaces nulles ou négatives :",
      (salles["surface_m2"] <= 0).sum())

print("Nombre de coachs négatif :",
      (salles["nb_coachs"] < 0).sum())

print("Anciennetés d'ouverture négatives :",
      (salles["ouverte_depuis_mois"] < 0).sum())

# Vérifier que chaque salle apparaît une seule fois
print("Noms de salles répétés :",
      salles["salle"].duplicated().sum())


# ==========================================
# ÉTAPE 8C : Correspondance abonnés / salles
# ==========================================

print("\n=== VÉRIFICATION ABONNÉS / SALLES ===")

# Rechercher les abonnés dont la salle n'existe pas
salles_inconnues = abonnes[
    ~abonnes["salle"].isin(salles["salle"])
]

# Compter les abonnés concernés
print("Abonnés avec une salle inconnue :", len(salles_inconnues))

# Afficher les salles inconnues
print("\nListe des salles inconnues :")
print(salles_inconnues["salle"].unique())


# ==========================================
# ÉTAPE 9 : Analyse des résiliations
# ==========================================

print("\n=== RÉPARTITION DES RÉSILIATIONS ===")

# Compter les abonnés qui restent et ceux qui partent
repartition = abonnes["a_resilie"].value_counts()

print(repartition)

# Calculer le pourcentage de chaque groupe
pourcentages = (
    abonnes["a_resilie"].value_counts(normalize=True) * 100
)

print("\n=== POURCENTAGE DES RÉSILIATIONS ===")
print(pourcentages.round(2))


# ==========================================
# ÉTAPE 9B : Analyse du prix des abonnements
# ==========================================

print("\n=== PRIX ET RÉSILIATIONS ===")

# Calculer le prix moyen selon la résiliation
prix_moyen = abonnes.groupby("a_resilie")["prix_mensuel_eur"].mean()

print("\nPrix moyen selon la résiliation :")
print(prix_moyen)

# Calculer le prix médian selon la résiliation
prix_median = abonnes.groupby("a_resilie")["prix_mensuel_eur"].median()

print("\nPrix médian selon la résiliation :")
print(prix_median)


# ==========================================
# ÉTAPE 9B : Résiliations selon la formule
# ==========================================

print("\n=== RÉSILIATIONS PAR FORMULE ===")

# Transformer oui/non en 1/0
abonnes["resiliation_num"] = (
    abonnes["a_resilie"].map({"oui": 1, "non": 0})
)

# Regrouper les abonnés selon leur formule
taux_formule = (
    abonnes.groupby("formule")["resiliation_num"]
    .agg(["count", "sum", "mean"])
)

# Transformer les taux en pourcentages
taux_formule["mean"] = taux_formule["mean"] * 100

print(taux_formule.round(2))


# ==========================================
# ÉTAPE 9C : Créneaux horaires et résiliations
# ==========================================

print("\n=== RÉSILIATIONS SELON LES CRÉNEAUX ===")

# Regrouper les abonnés selon leur créneau habituel
taux_creneaux = (
    abonnes.groupby("creneau_habituel")["resiliation_num"]
    .agg(["count", "sum", "mean"])
)

# Convertir les taux en pourcentages
taux_creneaux["mean"] = taux_creneaux["mean"] * 100

# Afficher les résultats
print(taux_creneaux.round(2))




# ==========================================
# ÉTAPE 9D : Fréquentation et résiliations
# ==========================================

print("\n=== FRÉQUENTATION ET RÉSILIATIONS ===")

# Comparer le nombre moyen de visites
visites_moyennes = (
    abonnes.groupby("a_resilie")["visites_par_mois"].mean()
)

print("\nVisites moyennes selon la résiliation :")
print(visites_moyennes)

# Créer des groupes selon le nombre de visites
abonnes["groupe_visites"] = pd.cut(
    abonnes["visites_par_mois"],
    bins=[-1, 10, 20, float("inf")],
    labels=[
        "0 à 10 visites",
        "11 à 20 visites",
        "Plus de 20 visites"
    ]
)

# Calculer les taux de résiliation
taux_visites = (
    abonnes.groupby("groupe_visites", observed=True)["resiliation_num"]
    .agg(["count", "sum", "mean"])
)

# Convertir en pourcentages
taux_visites["mean"] = taux_visites["mean"] * 100

print("\n=== TAUX DE RÉSILIATION SELON LES VISITES ===")
print(taux_visites.round(2))



# ==========================================
# ÉTAPE 9D : Réclamations et résiliations
# ==========================================

print("\n=== RÉCLAMATIONS ET RÉSILIATIONS ===")

# Calculer le nombre moyen de réclamations
reclamations_moyennes = (
    abonnes.groupby("a_resilie")["reclamations"].mean()
)

print("\nNombre moyen de réclamations :")
print(reclamations_moyennes)

# Calculer le taux de résiliation selon les réclamations
taux_reclamations = (
    abonnes.groupby("reclamations")["resiliation_num"]
    .agg(["count", "sum", "mean"])
)

# Convertir le taux en pourcentage
taux_reclamations["mean"] = taux_reclamations["mean"] * 100

print("\n=== TAUX DE RÉSILIATION PAR NOMBRE DE RÉCLAMATIONS ===")
print(taux_reclamations.round(2))


# ==========================================
# ÉTAPE 9E : Distance et résiliations
# ==========================================

print("\n=== DISTANCE ET RÉSILIATIONS ===")

# Convertir les distances de mètres en kilomètres
abonnes["distance_km"] = (
    abonnes["distance_domicile_m"] / 1000
)

# Calculer la distance moyenne pour chaque groupe
distance_moyenne = (
    abonnes.groupby("a_resilie")["distance_km"].mean()
)

print("\nDistance moyenne selon la résiliation :")
print(distance_moyenne.round(2))

# Calculer la distance médiane pour chaque groupe
distance_mediane = (
    abonnes.groupby("a_resilie")["distance_km"].median()
)

print("\nDistance médiane selon la résiliation :")
print(distance_mediane.round(2))


# ÉTAPE 9E : Taux de résiliation selon la distance

print("\n=== TAUX DE RÉSILIATION SELON LA DISTANCE ===")

# Créer trois groupes de distance
abonnes["groupe_distance"] = pd.cut(
    abonnes["distance_km"],
    bins=[0, 3, 6, float("inf")],
    labels=["0 à 3 km", "3 à 6 km", "Plus de 6 km"],
    include_lowest=True
)

# Calculer les résiliations par groupe
taux_distance = (
    abonnes.groupby("groupe_distance", observed=True)["resiliation_num"]
    .agg(["count", "sum", "mean"])
)

# Convertir les taux en pourcentages
taux_distance["mean"] = taux_distance["mean"] * 100

print(taux_distance.round(2))

# ==========================================
# ÉTAPE 9E : Préparation de la régression logistique
# ==========================================

print("\n=== PRÉPARATION DE LA RÉGRESSION LOGISTIQUE ===")

# Sélectionner les variables explicatives
X = abonnes[
    [
        "distance_km",
        "prix_mensuel_eur",
        "visites_par_mois",
        "anciennete_mois"
    ]
]

# Sélectionner la variable à expliquer
y = abonnes["resiliation_num"]

# Vérifier les dimensions
print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)

# Vérifier les éventuelles valeurs manquantes
print("\nValeurs manquantes dans X :")
print(X.isnull().sum())

# Afficher les premières lignes
print("\nVariables explicatives :")
print(X.head())

print("\nVariable à expliquer :")
print(y.head())
import statsmodels.api as sm

X_modele = sm.add_constant(X)

modele = sm.Logit(y, X_modele)

resultats = modele.fit(disp=False)

print(resultats.summary())


# ==========================================
# ÉTAPE 11A : Exportation des données nettoyées
# ==========================================

print("\n=== EXPORT DES DONNÉES NETTOYÉES ===")

# Enregistrer les abonnés nettoyés
abonnes.to_csv(
    "FitZone_abonnes_nettoyes.csv",
    index=False
)

# Enregistrer les salles vérifiées
salles.to_csv(
    "FitZone_salles_nettoyees.csv",
    index=False
)

print("Fichier abonnés enregistré !")
print("Fichier salles enregistré !")

print("Nombre d'abonnés :", len(abonnes))
print("Nombre de salles :", len(salles))


# ==========================================
# ÉTAPE 11B : Vérification des CSV exportés
# ==========================================

print("\n=== VÉRIFICATION DES FICHIERS EXPORTÉS ===")

# Relire les deux fichiers enregistrés
abonnes_verif = pd.read_csv("FitZone_abonnes_nettoyes.csv")
salles_verif = pd.read_csv("FitZone_salles_nettoyees.csv")

# Vérifier leurs dimensions
print("Dimensions abonnés :", abonnes_verif.shape)
print("Dimensions salles :", salles_verif.shape)

# Vérifier les doublons
print("Doublons abonnés :", abonnes_verif.duplicated().sum())
print("Doublons salles :", salles_verif.duplicated().sum())

# Vérifier les valeurs manquantes
print("Valeurs manquantes abonnés :",
      abonnes_verif.isnull().sum().sum())

print("Valeurs manquantes salles :",
      salles_verif.isnull().sum().sum())














