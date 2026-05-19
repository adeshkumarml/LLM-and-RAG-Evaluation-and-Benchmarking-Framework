from src.utils.utils import config_loader
from sentence_transformers import SentenceTransformer

config = config_loader()
embedding_model = SentenceTransformer(config["embedding_model"])

def generate_embeddings(texts):
    embeddings = embedding_model.encode(texts, convert_to_numpy = True)
    return embeddings