import pandas as pd
import numpy as np
import re
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_fr = stopwords.words('french')
stop_fr += ['tres', 'très', 'peu', 'plus', 'tout', 'tous', 'comme', 'donc', 'alors', 'aussi']
lemmatiseur = WordNetLemmatizer()

def nettoyer_texte(texte):
    if not isinstance(texte, str):
        return ""
    texte = texte.lower()
    texte = re.sub(r"http\S+|www.\S+", "", texte)
    texte = re.sub(r"@\w+|#\w+", "", texte)
    texte = re.sub(r"[^a-zàâäéèêëîïôöùûüç'\-]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    if len(texte) < 3:
        return ""
    mots = texte.split()
    return " ".join([lemmatiseur.lemmatize(m) for m in mots if m not in stop_fr and len(m) > 2])

print("="*50)
print("     TP5 — ENTRAINEMENT FINAL")
print("="*50)

df = pd.read_csv("data/sample_reviews.csv")
print(f"\n📊 Total : {len(df)} lignes")
print(f"Colonnes : {df.columns.tolist()}")
print(f"Répartition :\n{df['label'].value_counts()}")

df["texte_nettoye"] = df["review"].apply(nettoyer_texte)
df = df[df["texte_nettoye"] != ""].reset_index(drop=True)
print(f"\n✅ Après nettoyage : {len(df)} lignes exploitables")

X = df["texte_nettoye"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📦 Train : {len(X_train)} | Test : {len(X_test)}")

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
print(f"🔢 Vocabulaire : {len(tfidf.get_feature_names_out())} termes")

modele = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
modele.fit(X_train_vec, y_train)

y_pred = modele.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print(f"\n📈 PRECISION FINALE : {acc:.2%}")
print("-"*50)
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(modele, "models/modele_sentiment.joblib")
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
joblib.dump({"precision": acc, "nb_donnees": len(df)}, "models/metadonnees.joblib")

print(f"\n✅ MODELE SAUVEGARDE — Précision : {acc:.2%}")
print("="*50)
