# Recommandation d'articles de presse par LLM

L'objectif de ce projet est d'utiliser la technologie sous jacente des LLM pour concevoir un recommandeur d'articles à partir de mots clés.

Les différentes phases du projet :
- construction d'une base de données d'articles français via un web scraping journalier automatique avec Github Actions
- représentation vectorielle des titres d'article
- création d'une base de données vectorielles via Qdrant et mise à jour des données
- mise à disposition des résultats via API avec FastAPI
- création d'une application de recommandation Streamlit, plolty et scikit-learn (T-SNE pour créer un espace 3D pour visualiser les articles)
