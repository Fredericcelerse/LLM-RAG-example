from model import load_model, generate_answer
from embeddings import get_embedding_model, get_embeddings, semantic_search
from utils import extract_content, chunk_text
import numpy as np

def rag_answer(query, urls, hf_token):
    model, tokenizer = load_model(hf_token)
    embedding_model = get_embedding_model("jina-embeddings-v2")
    
    all_chunks = []
    all_embeddings = []
    
    for url in urls:
        content = extract_content(url)
        chunks = chunk_text(content)
        all_chunks.extend(chunks)
        all_embeddings.extend(get_embeddings(chunks, embedding_model))
    
    relevant_chunks = semantic_search(query, np.array(all_embeddings), all_chunks, embedding_model)
    context = "\n\n".join(relevant_chunks)
    
    answer = generate_answer(query, context, model, tokenizer)
    answer = answer.split("Answer:")[-1].strip()
    return answer

# Example usage
if __name__ == "__main__":
    hf_token = "your_hf_token_here"
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]
    query = "What are the main applications of artificial intelligence?"
    
    response = rag_answer(query, urls, hf_token)
    print(response)
