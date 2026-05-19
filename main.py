from src.retrieval.chunking import chunk_text
from src.retrieval.embedding import generate_embeddings
from src.orchestration.evaluator import run_evaluation
from src.utils.utils import logging
import json
import time

def main():
    with open("data/dataset.json", "r", encoding = "utf-8") as f:
        dataset = json.load(f)
    logging.info("Dataset loaded")

    all_chunks =[]
    for sample in dataset:
        chunks = chunk_text(sample["context"])
        all_chunks.extend(chunks)
    chunk_embeddings = generate_embeddings(all_chunks)
    logging.info("Embedding successful")
    logging.info("Starting evaluation")
    eval_results, aggregate_results = run_evaluation(dataset, all_chunks, chunk_embeddings)
    logging.info(f"Evaluation completed in {aggregate_results['runtime_secs']} seconds")

    print("Consolidated Results")
    for metric, value in aggregate_results.items():
        print(f"{metric}: {value:.2f}")


    with open("results/eval_results.json", "w", encoding = "utf-8") as f:
        json.dump(eval_results, f, indent = 4, ensure_ascii = False)
    logging.info("Individual sample results saved at results/eval_results.json")

    with open("results/agg_results.json", "w", encoding = "utf-8") as f:
        json.dump(aggregate_results, f, indent = 4)
    logging.info("Aggregate results saved at results/agg_results.json")

if __name__ == "__main__":
    main()