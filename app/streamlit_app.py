import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Chargement des ressources
@st.cache_resource
def charger_modele():
    modele = joblib.load("models/modele_sentiment.joblib")
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    return modele, tfidf

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r"[^a-zàâäéèêëîïôöùûüç']", " ", texte)
    mots = texte.split()
    stop_fr = stopwords.words('french')
    lemmatiseur = WordNetLemmatizer()
    mots_filtres = [lemmatiseur.lemmatize(m) for m in mots if m not in stop_fr and len(m) > 2]
    return " ".join(mots_filtres)

# Interface
st.title("🔍 Analyse de Sentiment")
st.subheader("Entraînement TP2")

avis = st.text_area("Ton avis :", placeholder="Écris ton avis ici...")

if st.button("Analyser"):
    if avis.strip():
        modele, tfidf = charger_modele()
        texte_nettoye = nettoyer_texte(avis)
        vecteur = tfidf.transform([texte_nettoye])
        prediction = modele.predict(vecteur)[0]
        proba = modele.predict_proba(vecteur).max()
        
        if prediction == "positive":
            st.success(f"😃 Positif — confiance : {proba:.1%}")
        else:
            st.error(f"😞 Négatif — confiance : {proba:.1%}")
    else:
        st.warning("Saisis un avis d'abord !")
