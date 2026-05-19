from src.utils.utils import config_loader
from src.retrieval.embedding import generate_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

config = config_loader()

def retrieve_top_k_chunks(query, chunk_embeddings, chunks, top_k = config["top_k"]):
    query_embedding = generate_embeddings([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_idx = np.argsort(similarities)[-top_k:][::-1]
    retrieved_chunks = [chunks[idx] for idx in top_idx]
    return retrieved_chunks

