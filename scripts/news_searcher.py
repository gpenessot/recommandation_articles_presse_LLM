from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import configparser

config=configparser.ConfigParser()
config.read('../config/config.cfg')

QDRANT_HOST=config['QDRANT']['host']
QDRANT_PORT=config['QDRANT']['port']
QDRANT_API_KEY=config['QDRANT']['qdrant_api_key']

class NewsSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer("moussaKam/barthez", device="cpu")
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(url=QDRANT_HOST, 
                                          port=QDRANT_PORT, 
                                          api_key=QDRANT_API_KEY)

    def search(self, text: str):
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
