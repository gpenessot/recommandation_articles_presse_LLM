# local tests
import configparser

# render server
# import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

config = configparser.ConfigParser()
config.read("../config/config.cfg")

# local test
QDRANT_HOST = config["QDRANT"]["host"]
QDRANT_PORT = config["QDRANT"]["port"]
QDRANT_API_KEY = config["QDRANT"]["qdrant_api_key"]

# QDRANT_HOST = os.getenv("host")
# QDRANT_PORT = os.getenv("port")
# QDRANT_API_KEY = os.getenv("qdrant_api_key")


class NewsSearcher:
    """
    A class for searching news articles using QdrantClient and SentenceTransformer.

    Attributes:
        collection_name (str): The name of the collection to search in.
        model (SentenceTransformer): The SentenceTransformer model used for encoding text 
        into vectors.
        qdrant_client (QdrantClient): The QdrantClient used for searching vectors in the 
        collection.

    Methods:
        search(text: str) -> List[Dict]:
            Searches for news articles similar to the given text.
            Args:
                text (str): The text to search for.
            Returns:
                List[Dict]: A list of dictionaries representing the search results. Each 
                dictionary contains the payload of a news article.
    """

    def __init__(self, collection_name):
        """
        A class for searching news articles using QdrantClient and SentenceTransformer.

        Attributes:
            collection_name (str): The name of the collection to search in.
            model (SentenceTransformer): The SentenceTransformer model used for encoding 
            text into vectors.
            qdrant_client (QdrantClient): The QdrantClient used for searching vectors in 
            the collection.
        """
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer("moussaKam/barthez", device="cpu")
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY
        )

    def search(self, text: str):
        """
        Searches for news articles similar to the given text.

        Args:
            text (str): The text to search for.

        Returns:
            List[Dict]: A list of dictionaries representing the search results. Each dictionary 
            contains the payload of a news article.
        """
        # Convert text query into vector
        vector = self.model.encode(text).tolist()

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,  # If you don't want any filters for now
            limit=5,  # 5 the most closest results is enough
        )
        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function you are interested in payload only
        payloads = [hit.payload for hit in search_result]
        return payloads


if __name__ == "__main__":
    neural_searcher = NewsSearcher(collection_name="articles_fr_newsapi")
    print(neural_searcher.search(text="Guerre en Ukraine"))
