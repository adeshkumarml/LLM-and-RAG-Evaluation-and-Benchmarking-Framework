from src.utils.utils import config_loader

config = config_loader()

def chunk_text(text, chunk_size = config["chunk_size"]):
    words = text.split()
    chunks =[]
    for i in range(0, len(words), chunk_size):
        chunk = words[i: i+chunk_size]
        chunks.append(" ".join(chunk))
    return chunks