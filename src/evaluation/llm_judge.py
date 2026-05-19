from src.utils.utils import config_loader, logging
from src.utils.openai_client import client
from src.evaluation.parser import parse_judge_scores

config = config_loader()
llm_model = config["llm_model"]
temperature = config["temperature"]

def llm_as_judge(question, prediction, ground_truths):
    ground_truths_text = "\n".join(ground_truths)
    prompt = f"""
        You are evaluating a QA system answer.
        Question: {question}
        Ground Truth Answer: {ground_truths_text}
        Model Answer: {prediction}
        Now, evaluate the Model Answer on the following: Correctness, Completeness and Fluency
        Give a score ranging from 1 to 5.

        Respond ONLY in this format:
        Correctness: <score>
        Completeness: <score>
        Fluency: <score>
        """
    
    try:
        response = client.chat.completions.create(
            model = llm_model,
            messages = [{"role": "user", "content": prompt}],
            temperature = temperature,
            max_tokens = 150,
            timeout = 30
        )
        judge_response = response.choices[0].message.content
        if judge_response is None:
            return {
                "correctness": 0,
                "completeness": 0,
                "fluency": 0
            }
        else:
            parsed_scores = parse_judge_scores(judge_response)
            return parsed_scores
    
    except Exception as e:
        logging.info(f"LLM Judge Error: {e}")