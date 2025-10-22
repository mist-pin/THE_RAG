from chromadb.utils import embedding_functions
from utils.constants import EMBEDDING_MODEL

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
