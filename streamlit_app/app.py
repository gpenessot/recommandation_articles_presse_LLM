# draft #1

import configparser

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

config = configparser.ConfigParser()
config.read("../config/config.cfg")

# local test
QDRANT_HOST = config["QDRANT"]["host"]
QDRANT_PORT = config["QDRANT"]["port"]
QDRANT_API_KEY = config["QDRANT"]["qdrant_api_key"]

# Qdrant Client
client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)


@st.cache_resource(ttl=24 * 3600, hash_funcs={"MyUnhashableClass": lambda _: None})
def load_model():
    return SentenceTransformer(
        "/home/gael/.cache/torch/sentence_transformers/moussaKam_barthez/"
    )


model = load_model()


# Function to encode text with the transformer model
def encode(query_text, model=model):
    encoder = model  # , device='cpu'
    return encoder.encode(query_text)


# Function to calculate distance between vectors
def calculate_distance(dataframe, query_vector):
    vector1 = np.array(dataframe["sentence_embedding"].tolist()).reshape(1, -1)
    vector2 = query_vector.reshape(1, -1)
    return cosine_similarity(vector1, vector2)[0][0]


def generate_item_sentence(item: pd.Series, text_columns=["title"]) -> str:
    """
    Process csv file to fit Qdrant requirements

    Parameters:
    - path_to_csv (str): The path to the csv file. Default is "../data/processed/articles.csv".

    Returns:
    - pd.DataFrame: The processed dataframe with the following columns:
        - newsId: The ID of the news article.
        - author: The author of the news article.
        - title: The title of the news article.
        - publishedAt: The publication date of the news article.
        - content: The content of the news article.
        - sentence: The concatenated sentence from the text columns.
        - sentence_embedding: The sentence embedding generated using the SentenceTransformer model.
    """
    return " ".join([item[column] for column in text_columns])


def prepare_csv_file(
    path_to_csv: str = "../data/processed/articles.csv",
) -> pd.DataFrame:
    """
    Process csv file to fit Qdrant requirements

    Parameters:
    - path_to_csv (str): The path to the csv file. Default is "../data/processed/articles.csv".

    Returns:
    - pd.DataFrame: The processed dataframe with the following columns:
        - newsId: The ID of the news article.
        - author: The author of the news article.
        - title: The title of the news article.
        - publishedAt: The publication date of the news article.
        - content: The content of the news article.
        - sentence: The concatenated sentence from the text columns.
        - sentence_embedding: The sentence embedding generated using the SentenceTransformer model.

    """
    df = pd.read_csv(path_to_csv, index_col=0, encoding="utf-8")
    df = df.reset_index()
    df.columns = ["newsId", "author", "title", "publishedAt", "content"]
    df["sentence"] = df.apply(generate_item_sentence, axis=1)
    df["sentence_embedding"] = df["sentence"].apply(encode)
    return df


# Streamlit App
st.title("LLM News searcher")
st.markdown(
    "Bienvenue, cette vous recommande les articles en fonction de votre recherche"
)

# Input text box
query_text = st.text_input("Entrez le thème qui vous intéresse :", "Automobile")

# Encode the query text
query_vector = encode(query_text)

articles = prepare_csv_file()

articles["distance"] = articles.apply(
    calculate_distance, query_vector=query_vector, axis=1
)

scaler = MinMaxScaler()
scaler.fit(articles[["distance"]])
articles["normalised"] = scaler.transform(articles[["distance"]])
results = articles.sort_values(by="distance", ascending=False).nlargest(
    3, columns="distance"
)

col1, col2 = st.columns([2, 2])

# Display 3 closest vectors
with col1:
    st.markdown("**Meilleurs articles :**")
    for index in range(results.shape[0]):
        st.markdown(f" * {results.iloc[index]['title']}")

with col2:
    st.markdown("**Détail de l'article :**")

    liste_titres = st.selectbox(
        "Quel article souhaitez-vous lire ?", tuple(results["title"].to_list()), index=0
    )

    st.markdown(articles[articles["title"] == liste_titres]["content"].values[0])

# Create a t-SNE model
tsne_model = TSNE(
    n_components=3, perplexity=15, random_state=42, init="random", learning_rate=200
)
tsne_embeddings = tsne_model.fit_transform(
    np.array(articles["sentence_embedding"].to_list())
)

# Create a DataFrame for visualisation
visualisation_data = pd.DataFrame(
    {
        "x": tsne_embeddings[:, 0],
        "y": tsne_embeddings[:, 1],
        "z": tsne_embeddings[:, 2],
        "title": articles["title"],
        "Similarité": articles["normalised"].round(3),
    }
)

# Create the scatter plot using Plotly Express
plot = px.scatter_3d(
    visualisation_data,
    x="x",
    y="y",
    z="z",
    color="Similarité",
    hover_name="title",
    color_continuous_scale="rainbow",
    opacity=0.7,
    title=f"Similarité à '{query_text}' visualisé avec t-SNE",
)

plot.update_layout(width=700, height=650)

plot.update_traces(textposition="top center")

st.plotly_chart(plot)
