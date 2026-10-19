import pandas as pd
import numpy as np
import re
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_fr = stopwords.words('french')
stop_fr += ['tre', 'tres', 'très', 'peu', 'plus', 'tout', 'tous', 'comme']
lemmatiseur = WordNetLemmatizer()

def nettoyer_texte(texte):
    texte = str(texte).lower()
    texte = re.sub(r"[^a-zàâäéèêëîïôöùûüç'\-]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    mots = texte.split()
    mots_filtres = [
        lemmatiseur.lemmatize(mot)
        for mot in mots
        if mot not in stop_fr and len(mot) > 2 and not mot.isdigit()
    ]
    return " ".join(mots_filtres)

print("=== TP3 : ENTRAINEMENT COMPLET ===")
df = pd.read_csv("data/sample_reviews.csv")

col_texte = "review"
col_cible = "label"

print(f"\nDonnees : {len(df)} lignes")
df["texte_nettoye"] = df[col_texte].apply(nettoyer_texte)

X = df["texte_nettoye"]
y = df[col_cible]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train : {len(X_train)} | Test : {len(X_test)}")

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=1,
    max_df=1.0,
    sublinear_tf=True
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

modeles = {
    "Regression Logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Naive Bayes": MultinomialNB(),
    "SVM Lineaire": LinearSVC(max_iter=2000),
    "Foret Aleatoire": RandomForestClassifier(n_estimators=200, random_state=42)
}

resultats = {}
meilleur_nom = None
meilleur_score = 0
meilleur_modele = None

print("\n--- COMPARAISON DES MODELES ---")
for nom, modele in modeles.items():
    modele.fit(X_train_vec, y_train)
    y_pred = modele.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    resultats[nom] = acc
    print(f"{nom:22} | Precision : {acc:.2%}")
    
    if acc > meilleur_score:
        meilleur_score = acc
        meilleur_nom = nom
        meilleur_modele = modele

print(f"\n🏆 MEILLEUR MODELE : {meilleur_nom} — {meilleur_score:.2%}")
print("\nRapport detaille :")
print(classification_report(y_test, meilleur_modele.predict(X_test_vec)))

os.makedirs("models", exist_ok=True)
joblib.dump(meilleur_modele, "models/modele_sentiment.joblib")
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
print(f"\n✅ Sauvegarde : {meilleur_nom} — {meilleur_score:.2%}")
