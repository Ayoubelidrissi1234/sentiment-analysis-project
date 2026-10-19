import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Analyse Sentiment — TP5", page_icon="🎯", layout="wide")

@st.cache_resource
def charger_tout():
    return (
        joblib.load("models/modele_sentiment.joblib"),
        joblib.load("models/tfidf_vectorizer.joblib"),
        joblib.load("models/metadonnees.joblib")
    )

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_fr = stopwords.words('french')
stop_fr += ['tres', 'très', 'peu', 'plus', 'tout', 'tous', 'comme', 'donc', 'alors']
lemmatiseur = WordNetLemmatizer()

def nettoyer_texte(t):
    if not isinstance(t, str) or len(t.strip()) < 3:
        return ""
    t = t.lower()
    t = re.sub(r"http\S+|www.\S+|@\w+|#\w+", "", t)
    t = re.sub(r"[^a-zàâäéèêëîïôöùûüç'\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    mots = t.split()
    return " ".join([lemmatiseur.lemmatize(m) for m in mots if m not in stop_fr and len(m) > 2])

modele, tfidf, meta = charger_tout()
nom_modele = type(modele).__name__

with st.sidebar:
    st.title("📊 Infos")
    st.info(f"**Modèle : {nom_modele}**")
    st.metric("Précision", f"{meta['precision']:.1%}")
    st.caption(f"Données : {meta['nb_donnees']} avis")

st.title("🎯 Analyse de Sentiment — Version Finale")
st.divider()

avis = st.text_area("💬 Ton avis :", placeholder="Ex: Excellente qualité, livraison rapide...", height=150)
c1, c2, c3 = st.columns([3, 1, 1])
with c1: ok = st.button("🔮 Analyser", type="primary", use_container_width=True)
with c2: p = st.button("😊 Positif", use_container_width=True)
with c3: n = st.button("😞 Négatif", use_container_width=True)

if p: avis = "Excellente qualité, je recommande !"; st.rerun()
if n: avis = "Déçu, ne fonctionne pas."; st.rerun()

if ok:
    if len(avis.strip()) < 5:
        st.warning("⚠️ Texte trop court.")
    else:
        propre = nettoyer_texte(avis)
        if not propre:
            st.warning("⚠️ Texte non exploitable.")
        else:
            vect = tfidf.transform([propre])
            pred = modele.predict(vect)[0]
            conf = f"{max(modele.predict_proba(vect)[0]):.1%}" if hasattr(modele, 'predict_proba') else "—"
            
            st.divider()
            if str(pred).lower() in ["positive", "pos", "1"]:
                st.success("## 😃 SENTIMENT POSITIF")
            else:
                st.error("## 😞 SENTIMENT NÉGATIF")
            st.info(f"Confiance : {conf} | Précision : {meta['precision']:.1%}")
            with st.expander("Texte analysé"): st.code(propre)

st.divider()
st.caption("TP5 — Projet Final")
