import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="Analyse de Sentiment — TP4",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner="Chargement du modèle...")
def charger_modele():
    try:
        modele = joblib.load("models/modele_sentiment.joblib")
        tfidf = joblib.load("models/tfidf_vectorizer.joblib")
        return modele, tfidf
    except FileNotFoundError as e:
        st.error(f"❌ Fichier manquant : {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {e}")
        st.stop()

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_fr = stopwords.words('french')
stop_fr += ['tres', 'très', 'peu', 'plus', 'tout', 'tous', 'comme', 'donc', 'alors']
lemmatiseur = WordNetLemmatizer()

def nettoyer_texte(texte):
    if not texte or not isinstance(texte, str):
        return ""
    texte = texte.lower()
    texte = re.sub(r"[^a-zàâäéèêëîïôöùûüç'\-]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    if len(texte) < 2:
        return ""
    mots = texte.split()
    mots_filtres = [
        lemmatiseur.lemmatize(mot)
        for mot in mots
        if mot not in stop_fr and len(mot) > 2 and not mot.isdigit()
    ]
    return " ".join(mots_filtres)

modele, tfidf = charger_modele()
nom_modele = type(modele).__name__

with st.sidebar:
    st.title("ℹ️ Informations")
    st.subheader("Modèle utilisé")
    st.info(f"**{nom_modele}**")
    st.subheader("Données")
    st.caption("Jeu d'entraînement : échantillon de 30 avis")
    st.warning("⚠️ Précision limitée — données insuffisantes")

st.title("🔍 Analyse de Sentiment — TP4")
st.markdown("### Détection d'avis positifs ou négatifs en français")
st.divider()

avis = st.text_area(
    "Ton avis :",
    placeholder="Ex: Excellente qualité, livraison rapide !",
    height=150,
    max_chars=1000
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    analyser = st.button("🔮 Analyser", type="primary", use_container_width=True)
with col2:
    exemple_pos = st.button("😊 Positif", use_container_width=True)
with col3:
    exemple_neg = st.button("😞 Négatif", use_container_width=True)

if exemple_pos:
    avis = "Je suis très satisfait, la qualité est excellente !"
    st.rerun()
if exemple_neg:
    avis = "Produit cassé, je ne recommande pas."
    st.rerun()

if analyser:
    if not avis or len(avis.strip()) < 3:
        st.warning("⚠️ Saisis un texte plus long.")
    else:
        with st.spinner("Analyse..."):
            texte_nettoye = nettoyer_texte(avis)
            if not texte_nettoye:
                st.warning("⚠️ Texte non exploitable.")
            else:
                vecteur = tfidf.transform([texte_nettoye])
                prediction = modele.predict(vecteur)[0]
                resultat = str(prediction).strip().lower()
                
                st.divider()
                confiance = None
                if hasattr(modele, 'predict_proba'):
                    proba = modele.predict_proba(vecteur)[0]
                    confiance = f"{max(proba):.1%}"
                
                col_res, col_info = st.columns([2, 1])
                with col_res:
                    if resultat in ["positive", "pos", "1"]:
                        st.success("## 😃 SENTIMENT POSITIF")
                    else:
                        st.error("## 😞 SENTIMENT NÉGATIF")
                with col_info:
                    if confiance:
                        st.metric("Confiance", confiance)
                    st.caption(f"Modèle : {nom_modele}")
                
                with st.expander("Texte traité"):
                    st.code(texte_nettoye, language=None)

st.divider()
st.caption("Projet TALN — TP4")
