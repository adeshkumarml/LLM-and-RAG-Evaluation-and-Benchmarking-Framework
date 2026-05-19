from src.utils.utils import logging, config_loader
from src.utils.openai_client import client
from src.retrieval.retriever import retrieve_top_k_chunks

config = config_loader()

llm_model = config["llm_model"]
temperature = config["temperature"]
top_k = config["top_k"]

def rag_generate_answer(question, chunks, chunk_embeddings):
    retrieved_chunks = retrieve_top_k_chunks(query = question, chunk_embeddings = chunk_embeddings, chunks = chunks, top_k = top_k)
    retrieved_context = "\n".join(retrieved_chunks)
    prompt = f"""
    Use ONLY and ONLY the provided context to answer the question. If the answer cannot be found in the context, be honest and say "I don't know."
    Context:
    {retrieved_context}
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
            return "", retrieved_chunks
        return answer.strip(), retrieved_chunks
    
    except Exception as e:
        logging.info(f"RAG pipeline error: {e}")
