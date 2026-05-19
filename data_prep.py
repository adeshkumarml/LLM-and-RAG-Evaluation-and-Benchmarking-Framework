from src.utils.utils import logging
from datasets import load_dataset
import json

dataset = load_dataset("squad_v2", split = "train")

dataset = dataset.filter(lambda x: len(x["answers"]["text"]) > 0)
dataset = dataset.shuffle(seed = 42).select(range(150))

processed_data = []

for item in dataset:
    processed_data.append({
        "id": item["id"],
        "context": item["context"],
        "question": item["question"],
        "answer": item["answers"]["text"] if item["answers"]["text"] else ""
    })

with open("data/dataset.json", "w") as f:
    json.dump(processed_data, f, indent=2)

logging.info(f"Saved {len(processed_data)} samples at 'data/dataset.json'.")
