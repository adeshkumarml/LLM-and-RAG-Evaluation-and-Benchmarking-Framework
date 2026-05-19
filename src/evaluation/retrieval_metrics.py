from src.evaluation.metrics import norm_text

def is_relevant(chunk, ground_truths):
    normalized_chunk = norm_text(chunk)   
    normalized_groud_truths = [norm_text(gt) for gt in ground_truths]
    for gt in normalized_groud_truths:
        if gt in normalized_chunk:
            return True
    return False


def precision_at_k(retrieved_chunks, ground_truths):
    relevance_count = 0
    for chunk in retrieved_chunks:
        if is_relevant(chunk, ground_truths):
            relevance_count += 1
    return relevance_count/len(retrieved_chunks)


def recall_at_k(retrieved_chunks, ground_truths):
    for chunk in retrieved_chunks:
        if is_relevant(chunk, ground_truths):
            return 1
    return 0


def mrr(retrieved_chunks, ground_truths):
    for i, chunk in enumerate (retrieved_chunks, 1):
        if is_relevant(chunk, ground_truths):
            return 1/i
    return 0
      