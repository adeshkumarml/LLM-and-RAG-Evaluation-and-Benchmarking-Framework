import string
from sklearn.metrics.pairwise import cosine_similarity
from src.retrieval.embedding import generate_embeddings

def norm_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

def exact_match(prediction, ground_truths):
    norm_pred = norm_text(prediction)
    norm_ground_truths = [norm_text(gt) for gt in ground_truths]
    return int(norm_pred in norm_ground_truths)

def semantic_sim(prediction, ground_truths):
    pred_embedding = generate_embeddings([prediction])
    similarities = []
    
    for gt in ground_truths:
        gt_embedding = generate_embeddings([gt])
        similarity = cosine_similarity(pred_embedding, gt_embedding)[0][0]
        similarities.append(similarity)
    
    return float(max(similarities))