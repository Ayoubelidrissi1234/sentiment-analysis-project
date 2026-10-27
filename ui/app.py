import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ---------- Configuration de la page (toujours en premier) ----------
st.set_page_config(page_title="Classifieur intelligent", layout="wide")
st.title("Application intelligente : classification interactive")

# ---------- Barre latérale : les réglages de l'utilisateur ----------
st.sidebar.header("Configuration")
jeu = st.sidebar.selectbox("Jeu de données", ["Iris", "Wine"])
algo = st.sidebar.selectbox("Modèle", ["Random Forest", "Régression logistique", "KNN"])
test_size = st.sidebar.slider("Part de test", 0.1, 0.5, 0.2, 0.05)

params = {}
if algo == "Random Forest":
    params["n"] = st.sidebar.slider("Nombre d'arbres", 10, 300, 100, 10)
    params["d"] = st.sidebar.slider("Profondeur max", 1, 20, 5)
elif algo == "KNN":
    params["k"] = st.sidebar.slider("Nombre de voisins (k)", 1, 25, 5)

if st.sidebar.button("Vider le cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# ---------- Données : mises en cache ----------
@st.cache_data
def charger(nom):
    data = load_iris() if nom == "Iris" else load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="classe")
    return X, y, list(data.target_names)

# ---------- Modèle : mis en cache ----------
@st.cache_resource
def entrainer(jeu, algo, test_size, n=None, d=None, k=None):
    X, y, _ = charger(jeu)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    if algo == "Random Forest":
        m = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
    elif algo == "Régression logistique":
        m = LogisticRegression(max_iter=2000)
    else:
        m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_tr, y_tr)
    return m, X_te, y_te

X, y, noms_classes = charger(jeu)
modele, X_te, y_te = entrainer(jeu, algo, test_size,
                               params.get("n"), params.get("d"), params.get("k"))
y_pred = modele.predict(X_te)
acc = accuracy_score(y_te, y_pred)
df_aff = X.assign(classe=y.map(dict(enumerate(noms_classes))))

# ---------- Trois onglets ----------
tab1, tab2, tab3 = st.tabs(["Données", "Performance", "Prédiction"])

with tab1:
    st.subheader(f"Jeu de données {jeu}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observations", X.shape[0])
    c2.metric("Variables", X.shape[1])
    c3.metric("Classes", len(noms_classes))
    st.dataframe(df_aff.head(20), use_container_width=True)

    st.subheader("Nuage de points")
    col_x = st.selectbox("Axe X", X.columns, index=0)
    col_y = st.selectbox("Axe Y", X.columns, index=1)
    st.plotly_chart(px.scatter(df_aff, x=col_x, y=col_y, color="classe", height=450),
                    use_container_width=True)

with tab2:
    st.subheader(f"Modèle : {algo}")
    st.metric("Exactitude (test)", f"{acc:.1%}")

    cm = confusion_matrix(y_te, y_pred)
    st.plotly_chart(px.imshow(cm, text_auto=True, x=noms_classes, y=noms_classes,
                              labels=dict(x="Prédit", y="Réel"),
                              color_continuous_scale="Blues",
                              title="Matrice de confusion"),
                    use_container_width=True)

    with st.expander("Rapport de classification détaillé"):
        st.text(classification_report(y_te, y_pred, target_names=noms_classes))

    if algo == "Random Forest":
        imp = pd.Series(modele.feature_importances_, index=X.columns).sort_values()
        st.plotly_chart(px.bar(imp, orientation="h", title="Importance des variables"),
                        use_container_width=True)

with tab3:
    st.subheader("Prédire une nouvelle observation")
    cols = st.columns(2)
    valeurs = {}
    for i, c in enumerate(X.columns):
        valeurs[c] = cols[i % 2].slider(
            c, float(X[c].min()), float(X[c].max()), float(X[c].mean()), key=f"{jeu}_{c}")

    obs = pd.DataFrame([valeurs])
    pred = modele.predict(obs)[0]
    st.success(f"Classe prédite : **{noms_classes[pred]}**")

    if hasattr(modele, "predict_proba"):
        proba = pd.Series(modele.predict_proba(obs)[0], index=noms_classes)
        st.plotly_chart(px.bar(proba, title="Probabilités par classe", range_y=[0, 1]),
                        use_container_width=True)
