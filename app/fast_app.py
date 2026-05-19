import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from fastapi import FastAPI, UploadFile, File, HTTPException
app = FastAPI()

from src.retrieval.chunking import chunk_text
from src.retrieval.embedding import generate_embeddings
from src.orchestration.evaluator import run_evaluation
from src.utils.utils import logging
from pydantic import BaseModel, ValidationError
from typing import List
import json

class EvalSample(BaseModel):
    question: str
    answer: List[str]

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/evaluate")
async def evaluate(corpus_file: UploadFile = File(...), dataset_file: UploadFile = File(...)):
    try:
        logging.info("Received evaluation request")
        corpus_content = await corpus_file.read()
        corpus_text = corpus_content.decode("utf-8")
        logging.info("Corpus loaded successfully")

        dataset_content = await dataset_file.read()
        dataset = json.loads(dataset_content.decode("utf-8"))

        validated_dataset = [EvalSample(**sample).model_dump() for sample in dataset]
        logging.info("Dataset schema validated")

        if len(validated_dataset) > 10:
            raise HTTPException(status_code = 400, detail = "Maximum 10 samples allowed")
        logging.info(f"{len(validated_dataset)} samples received")

        all_chunks = chunk_text(corpus_text)
        chunk_embeddings = generate_embeddings(all_chunks)
        eval_results, aggregate_results = run_evaluation(dataset, all_chunks, chunk_embeddings)
        logging.info("Evaluation completed")

        return {
            "aggregate_results": aggregate_results,
            "evaluation_results": eval_results
        }
    
    except ValidationError as e:
        logging.error(f"Pydantic validation failed: {e}")
        raise HTTPException(status_code = 400, detail = e.errors())
    
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code = 500, detail = str(e)) 