from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import numpy as np
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read("../config/config.cfg")

model_name = "moussaKam/barthez"
encoder = SentenceTransformer(model_name_or_path=model_name)

QDRANT_HOST = config["QDRANT"]["host"]
QDRANT_PORT = config["QDRANT"]["port"]
QDRANT_API_KEY = config["QDRANT"]["qdrant_api_key"]
CHUNK_SIZE = 500

client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)


def generate_item_sentence(item: pd.Series, text_columns=["title"]) -> str:
    return " ".join([item[column] for column in text_columns])


def prepare_csv_file(
    path_to_csv: str = "../data/processed/articles.csv",
) -> pd.DataFrame:
    """Process csv file to fit Qdrant requirements"""
    df = pd.read_csv(path_to_csv, index_col=0, encoding="utf-8")
    df = df.reset_index()
    df.columns = ["newsId", "author", "title", "publishedAt", "content"]
    df["sentence"] = df.apply(generate_item_sentence, axis=1)
    df["sentence_embedding"] = df["sentence"].apply(encoder.encode)
    return df


client.recreate_collection(
    collection_name="articles_fr_newsapi",
    vectors_config=VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=Distance.COSINE,
    ),
)


def create_vector_point(item: pd.Series) -> PointStruct:
    """Turn vectors into PointStruct"""
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
    df = prepare_csv_file()
    metadata_columns = df.drop(
        ["newsId", "sentence", "sentence_embedding"], axis=1
    ).columns

    points = df.apply(create_vector_point, axis=1).tolist()
    n_chunks = np.ceil(len(points) / CHUNK_SIZE)

    for i, points_chunk in enumerate(np.array_split(points, n_chunks)):
        client.upsert(
            collection_name="articles_fr_newsapi",
            wait=True,
            points=points_chunk.tolist(),
        )
