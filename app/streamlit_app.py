import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Analyse de Sentiment", page_icon="🔍", layout="centered")

@st.cache_resource
def charger_modele():
    modele = joblib.load("models/modele_sentiment.joblib")
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    return modele, tfidf

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_fr = stopwords.words('french')
lemmatiseur = WordNetLemmatizer()

def nettoyer_texte(texte):
    texte = str(texte).lower()
    texte = re.sub(r"[^a-zàâäéèêëîïôöùûüç'\-]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    if not texte:
        return "aucun"
    mots = texte.split()
    mots_filtres = [lemmatiseur.lemmatize(m) for m in mots if mot not in stop_fr and len(mot) > 2]
    return " ".join(mots_filtres)

# Interface
st.title("🔍 Analyse de Sentiment — TP3")
st.markdown("Détection automatique d'avis positifs ou négatifs")

avis = st.text_area("Ton avis :", placeholder="Ex: Produit excellent, livraison rapide...", height=120)

col1, col2 = st.columns([1, 1])
with col1:
    analyser = st.button("🔮 Analyser", type="primary", use_container_width=True)
with col2:
    exemple = st.button("💡 Exemple", use_container_width=True)

if exemple:
    st.session_state.avis = "Je suis tres satisfait, la qualite est au rendez-vous !"
    st.rerun()

if analyser:
    if avis and avis.strip():
        modele, tfidf = charger_modele()
        texte_nettoye = nettoyer_texte(avis)
        vecteur = tfidf.transform([texte_nettoye])
        prediction = modele.predict(vecteur)[0]
        
        st.divider()
        
        # Affichage confiance si disponible
        if hasattr(modele, 'predict_proba'):
            proba = max(modele.predict_proba(vecteur)[0])
            st.info(f"Confiance : {proba:.1%}")
        else:
            st.info("Confiance : non disponible (SVM)")
        
        resultat = str(prediction).strip().lower()
        if resultat in ["positive", "pos", "1"]:
            st.success("### 😃 SENTIMENT POSITIF")
        else:
            st.error("### 😞 SENTIMENT NEGATIF")
            
        st.caption("⚠️ Modele limite — jeu de donnees reduit (30 lignes)")
    else:
        st.warning("⚠️ Saisis un avis d'abord !")

st.divider()
st.caption("Application TP3 — Modele : Regression Logistique")
