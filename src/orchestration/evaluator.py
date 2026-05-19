from src.pipelines.baseline import baseline_generate_answer
from src.pipelines.rag import rag_generate_answer
from src.evaluation.metrics import exact_match, semantic_sim
from src.evaluation.retrieval_metrics import precision_at_k, recall_at_k, mrr
from src.evaluation.llm_judge import llm_as_judge
from src.utils.utils import logging, config_loader
import time
import json

config = config_loader()

def run_evaluation(dataset, all_chunks, chunk_embeddings):
    eval_results = []

    start_time = time.time()
    for i, sample in enumerate(dataset, 1):
            try:
                logging.info(f"Processing sample {i}/{len(dataset)}")

                question = sample["question"]
                ground_truths = sample["answer"]

                baseline_answer = baseline_generate_answer(question)
                rag_answer, retrieved_chunks = (rag_generate_answer(question, all_chunks, chunk_embeddings))

                baseline_em = exact_match(baseline_answer, ground_truths)
                rag_em = exact_match(rag_answer, ground_truths)
                baseline_semantic = semantic_sim(baseline_answer, ground_truths)
                rag_semantic = semantic_sim(rag_answer, ground_truths)

                precision = precision_at_k(retrieved_chunks, ground_truths)
                recall = recall_at_k(retrieved_chunks, ground_truths)
                mean_reciprocal = mrr(retrieved_chunks, ground_truths)

                baseline_judge_scores = llm_as_judge(question, baseline_answer, ground_truths)
                rag_judge_scores = llm_as_judge(question, rag_answer, ground_truths)

                eval_results.append(
                    {
                        "sample_id": f"{i:03}",
                        "question": question,
                        "ground_truths": ground_truths,
                        "baseline_answer": baseline_answer,
                        "rag_answer": rag_answer,
                        "retrieved_chunks": retrieved_chunks,
                        "baseline_em": baseline_em,
                        "rag_em": rag_em,
                        "baseline_semantic": baseline_semantic,
                        "rag_semantic": rag_semantic,
                        "precision_at_k": precision,
                        "recall_at_k": recall,
                        "mrr": mean_reciprocal,
                        "baseline_judge_scores": baseline_judge_scores,
                        "rag_judge_scores": rag_judge_scores
                    }
                )
            except Exception as e:
                logging.info(f"Sample failed {i}: {e}")
            if(i % 10 == 0):
                with open("results/intermediate_eval_results.json", "w", encoding = "utf-8") as f:
                    json.dump(eval_results, f, indent = 4, ensure_ascii = False)
                logging.info(f"Intermediate results at sample {i} has been saved at results/intermediate_eval_results.json")
    elapsed_time = time.time() - start_time

    logging.info("Aggregating results")
    aggregate_results = {}

    aggregate_results["runtime_secs"] = elapsed_time
    aggregate_results["num_samples"] = len(eval_results)
    aggregate_results["embedding_model"] = config["embedding_model"]
    aggregate_results["llm_model"] = config["llm_model"]
    aggregate_results["top_k"] = config["top_k"]    

    aggregate_results["baseline_em"] = sum(result["baseline_em"] for result in eval_results)/len(eval_results)
    aggregate_results["rag_em"] = sum(result["rag_em"] for result in eval_results)/len(eval_results)

    aggregate_results["baseline_semantic"] = sum(result["baseline_semantic"] for result in eval_results)/len(eval_results)
    aggregate_results["rag_semantic"] = sum(result["rag_semantic"] for result in eval_results)/len(eval_results)

    aggregate_results["precision_at_k"] = sum(result["precision_at_k"] for result in eval_results)/len(eval_results)
    aggregate_results["recall_at_k"] = sum(result["recall_at_k"] for result in eval_results)/len(eval_results)
    aggregate_results["mrr"] = sum(result["mrr"] for result in eval_results)/len(eval_results)

    aggregate_results["baseline_correctness"] = sum(result["baseline_judge_scores"]["correctness"] for result in eval_results)/len(eval_results)
    aggregate_results["baseline_completeness"] = sum(result["baseline_judge_scores"]["completeness"] for result in eval_results)/len(eval_results)
    aggregate_results["baseline_fluency"] = sum(result["baseline_judge_scores"]["fluency"] for result in eval_results)/len(eval_results)

    aggregate_results["rag_correctness"] = sum(result["rag_judge_scores"]["correctness"] for result in eval_results)/len(eval_results)
    aggregate_results["rag_completeness"] = sum(result["rag_judge_scores"]["completeness"] for result in eval_results)/len(eval_results)
    aggregate_results["rag_fluency"] = sum(result["rag_judge_scores"]["fluency"] for result in eval_results)/len(eval_results)

    return eval_results, aggregate_results