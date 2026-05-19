from src.utils.utils import logging, config_loader
from src.utils.openai_client import client

config = config_loader()
llm_model = config["llm_model"]
temperature = config["temperature"]

def baseline_generate_answer(question):
    prompt = f"""
    Answer the following question briefly and factually. If you are unsure, be honest and say "I don't know."
    Question: {question}"""
    try:
        response = client.chat.completions.create(
            model = llm_model,
            messages = [{"role": "user", "content": prompt}],
            temperature = temperature,
            max_tokens = 150,
            timeout = 30
        )
        answer = response.choices[0].message.content.strip()
        if answer is None:
            return ""
        return answer.strip()
    
    except Exception as e:
        logging.info(f"Baseline pipeline error: {e}")