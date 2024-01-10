# Recommandation d'articles de presse par LLM

L'objectif de ce projet est d'utiliser la technologie sous jacente des LLM pour concevoir un recommandeur d'articles à partir de mots clés.

Les différentes phases du projet :
- construction d'une base de données d'articles français
- représentation vectorielle des titres d'article
- création d'une base de données vectorielles via Qdrant et mise à jour des données
- mise à disposition des résultats via API avec FastAPI
- création d'une application de recommandation Streamlit, plolty et scikit-learn (TSN-E) pour la dataviz
