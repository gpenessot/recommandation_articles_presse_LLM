import configparser
import os

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


MODEL_NAME = "moussaKam/barthez"
CHUNK_SIZE = 500


def load_model(model=MODEL_NAME):
    """
    Initializes and returns an encoder.
    Reads model's name and load it.

    Returns:
    An encoder
    """
    encoder = SentenceTransformer(model_name_or_path=model)
    return encoder


def qdrant_client() -> QdrantClient:
    """
    Initializes and returns a QdrantClient instance.
    Reads configuration settings either from environment variables (if running in a GitHub Actions environment)
    or from a configuration file.

    Returns:
    A QdrantClient instance configured with the specified host, port, and API key.
    """
    if os.environ.get("GH_ACTIONS") == "true":
        qdrant_host = os.environ.get("QDRANT_HOST")
        qdrant_port = os.environ.get("QDRANT_PORT")
        qdrant_api_key = os.environ.get("QDRANT_API_TOKEN")
    else:
        config = configparser.ConfigParser()
        config.read("../config/config.cfg")
        qdrant_host = config["QDRANT"]["host"]
        qdrant_port = config["QDRANT"]["port"]
        qdrant_api_key = config["QDRANT"]["qdrant_api_key"]

    client = QdrantClient(url=qdrant_host, port=qdrant_port, api_key=qdrant_api_key)
    return client


def generate_item_sentence(item: pd.Series, text_columns=["title"]) -> str:
    """
    Process csv file to fit Qdrant requirements

    Parameters:
    - item (pd.Series): a series
    - text_columns=["title"]

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
    encoder,
    path_to_csv: str = "/home/runner/work/recommandation_articles_presse_LLM/recommandation_articles_presse_LLM/data/processed/articles.csv",
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
    df = df.drop_duplicates(keep="last")
    df.columns = ["newsId", "author", "title", "publishedAt", "content"]
    df["sentence"] = df.apply(generate_item_sentence, axis=1)
    df["sentence_embedding"] = df["sentence"].apply(encoder.encode)
    return df


def create_vector_point(item: pd.Series) -> PointStruct:
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
    return PointStruct(
        id=item["newsId"],
        vector=item["sentence_embedding"].tolist(),
        payload={
            field: item[field]
            for field in metadata_columns
            if (str(item[field]) not in ["None", "nan"])
        },
    )


if __name__ == "__main__":
    encoder = load_model()
    articles = prepare_csv_file(encoder)
    metadata_columns = articles.drop(
        ["newsId", "sentence", "sentence_embedding"], axis=1
    ).columns

    points = articles.apply(create_vector_point, axis=1).tolist()
    n_chunks = np.ceil(len(points) / CHUNK_SIZE)

    client = qdrant_client()
    client.recreate_collection(
        collection_name="articles_fr_newsapi",
        vectors_config=VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=Distance.COSINE,
        ),
    )

    for i, points_chunk in enumerate(np.array_split(points, n_chunks)):
        client.upsert(
            collection_name="articles_fr_newsapi",
            wait=True,
            points=points_chunk.tolist(),
        )
