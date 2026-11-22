import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

SEED = 42
os.makedirs("figures", exist_ok=True)



FICHIER_ABONNES = "FitZone_abonnes.csv"   # à remplacer par le fichier nettoyé d'Amadou
FICHIER_SALLES = "FitZone_salles.csv"


CIBLE = None

abonnes = pd.read_csv(FICHIER_ABONNES)
salles = pd.read_csv(FICHIER_SALLES)
print(abonnes.shape, salles.shape)
print(abonnes.head())

abonnes = abonnes.drop_duplicates().copy()
if "abonne_id" in abonnes.columns:
    abonnes = abonnes.drop_duplicates(subset="abonne_id", keep="first")
abonnes = abonnes.reset_index(drop=True)

# Valeurs incohérentes : ancienneté négative, prix nul ou négatif -> traitées comme manquantes
# (elles seront remplacées par la médiane dans le pipeline, sans fuite d'information)
if "anciennete_mois" in abonnes.columns:
    abonnes.loc[abonnes["anciennete_mois"] < 0, "anciennete_mois"] = np.nan
if "prix_mensuel_eur" in abonnes.columns:
    abonnes.loc[abonnes["prix_mensuel_eur"] <= 0, "prix_mensuel_eur"] = np.nan

# Détection de la colonne cible
if CIBLE is None:
    candidats = [c for c in abonnes.columns
                 if any(k in c.lower() for k in ["resil", "churn", "parti", "quitt"])]
    print("Colonnes candidates pour la cible :", candidats)
    if not candidats:
        raise ValueError("Précise la colonne cible dans la variable CIBLE.")
    CIBLE = candidats[0]
print("Cible utilisée :", CIBLE)

# Passage de la cible en 0/1
y_brut = abonnes[CIBLE]
if pd.api.types.is_numeric_dtype(y_brut):
    y = (y_brut > 0).astype(int)
else:
    y = y_brut.astype(str).str.strip().str.lower().isin(["1", "oui", "yes", "true", "resilie", "résilié", "vrai"]).astype(int)
abonnes = abonnes[y.notna()].copy()
print("Taux de résiliation global :", round(y.mean(), 3))

cles_communes = [c for c in abonnes.columns if c in salles.columns and c != "abonne_id"]
if cles_communes:
    cle = cles_communes[0]
    donnees = abonnes.merge(salles.drop_duplicates(subset=cle), on=cle, how="left")
    print("Jointure sur :", cle)
else:
    donnees = abonnes.copy()

a_exclure = {CIBLE, "abonne_id"} | {c for c in donnees.columns
                                    if "resil" in c.lower() and c != CIBLE}
variables = [c for c in donnees.columns if c not in a_exclure]
X = donnees[variables]
assert len(donnees) == len(y), "La jointure a changé le nombre d'abonnés"
y = y.reset_index(drop=True)

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
print("Numériques :", num_cols)
print("Catégorielles :", cat_cols)

taille_test = 400 if len(X) >= 800 else 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=taille_test, stratify=y, random_state=SEED)
print("Train :", X_train.shape, "| Test :", X_test.shape)

pretraitement = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

modeles = {
    "Régression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = {}
for nom, m in modeles.items():
    pipe = Pipeline([("prep", pretraitement), ("clf", m)])
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    scores[nom] = s.mean()
    print(f"{nom} : AUC moyenne = {s.mean():.3f} (+/- {s.std():.3f})")

meilleur = max(scores, key=scores.get)
print("\nModèle retenu :", meilleur)

modele = Pipeline([("prep", pretraitement), ("clf", modeles[meilleur])])
modele.fit(X_train, y_train)

proba = modele.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print("AUC test :", round(roc_auc_score(y_test, proba), 3))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=3))

imp = permutation_importance(modele, X_test, y_test, scoring="roc_auc",
                             n_repeats=10, random_state=SEED)
importances = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
importances.plot(kind="barh", figsize=(7, 4), color="steelblue")
plt.title("Importance des variables (baisse d'AUC si mélangée)")
plt.tight_layout()
plt.savefig("figures/importance_variables.png", dpi=150)
plt.show()
print(importances.sort_values(ascending=False))

resultat = X_test.copy()
if "abonne_id" in donnees.columns:
    resultat.insert(0, "abonne_id", donnees.loc[X_test.index, "abonne_id"].values)
resultat["resilie_reel"] = y_test.values
resultat["risque"] = proba
resultat = resultat.sort_values("risque", ascending=False).reset_index(drop=True)
resultat["rang"] = resultat.index + 1

priorite_80 = resultat.head(80)
priorite_80.to_csv("abonnes_a_appeler_80.csv", index=False)
print(priorite_80.head(10))


n = len(resultat)
taux_base = resultat["resilie_reel"].mean()
total_resilies = resultat["resilie_reel"].sum()

lignes = []
for k in (80, 160):
    top = resultat.head(k)
    captures = top["resilie_reel"].sum()
    precision = captures / k
    rappel = captures / total_resilies
    lift = precision / taux_base
    attendu_hasard = k * taux_base
    lignes.append({"appels": k, "résiliations captées": int(captures),
                   "précision": round(precision, 3), "rappel": round(rappel, 3),
                   "lift": round(lift, 2),
                   "captées au hasard (attendu)": round(attendu_hasard, 1)})
comparaison = pd.DataFrame(lignes)
print(f"Abonnés : {n} | résiliations : {int(total_resilies)} | taux de base : {taux_base:.3f}")
print(comparaison.to_string(index=False))
comparaison.to_csv("comparaison_80_160.csv", index=False)


resultat["resilies_cumules"] = resultat["resilie_reel"].cumsum()
part_contactes = resultat["rang"] / n * 100
gains = resultat["resilies_cumules"] / total_resilies * 100
lift_courbe = (resultat["resilies_cumules"] / resultat["rang"]) / taux_base

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(part_contactes, gains, label="Modèle", color="steelblue", lw=2)
axes[0].plot([0, 100], [0, 100], "--", color="gray", label="Hasard")
for k, c in ((80, "green"), (160, "orange")):
    axes[0].axvline(k / n * 100, color=c, ls=":", label=f"{k} appels")
axes[0].set_xlabel("% d'abonnés contactés")
axes[0].set_ylabel("% des résiliations captées")
axes[0].set_title("Courbe de gains cumulés")
axes[0].legend()

axes[1].plot(part_contactes, lift_courbe, color="steelblue", lw=2)
axes[1].axhline(1, ls="--", color="gray", label="Hasard (lift = 1)")
for k, c in ((80, "green"), (160, "orange")):
    axes[1].axvline(k / n * 100, color=c, ls=":", label=f"{k} appels")
axes[1].set_xlabel("% d'abonnés contactés")
axes[1].set_ylabel("Lift")
axes[1].set_title("Courbe de Lift")
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/gains_lift.png", dpi=150)
plt.show()


