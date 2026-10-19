import pandas as pd
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

nltk.download('stopwords')
nltk.download('wordnet')

def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r"[^a-zàâäéèêëîïôöùûüç']", " ", texte)
    mots = texte.split()
    stop_fr = stopwords.words('french')
    lemmatiseur = WordNetLemmatizer()
    mots_filtres = [lemmatiseur.lemmatize(m) for m in mots if m not in stop_fr and len(m) > 2]
    return " ".join(mots_filtres)

print("Chargement des donnees...")
df = pd.read_csv("data/sample_reviews.csv")

# === NOMS CORRIGES ===
col_texte = "review"
col_cible = "label"

df["texte_nettoye"] = df[col_texte].apply(nettoyer_texte)

print("Decoupage train/test...")
X = df["texte_nettoye"]
y = df[col_cible]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Vectorisation TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print("Entrainement du modele...")
modele = LogisticRegression(max_iter=500, class_weight="balanced")
modele.fit(X_train_vec, y_train)

y_pred = modele.predict(X_test_vec)
print("\n=== RESULTATS ===")
print(classification_report(y_test, y_pred))
print(f"Precision globale : {accuracy_score(y_test, y_pred):.2%}")

os.makedirs("models", exist_ok=True)
joblib.dump(modele, "models/modele_sentiment.joblib")
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
print("\n✅ Modele et vectoriseur sauvegardes dans models/")
