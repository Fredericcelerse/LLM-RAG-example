from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import numpy as np

# Function to select the embedding model
def get_embedding_model(model_name):
    if model_name == "jina-embeddings-v2":
        return SentenceTransformer('jinaai/jina-embeddings-v2-base-en')
    elif model_name == "longformer":
        return AutoModel.from_pretrained("allenai/longformer-base-4096")
    elif model_name == "e5-large":
        return SentenceTransformer('intfloat/e5-large-v2')
    elif model_name == "all-MiniLM":
        return SentenceTransformer('all-MiniLM-L6-v2')
    else:
        raise ValueError(f"Model {model_name} not supported")

# Function to generate embeddings for a list of texts
def get_embeddings(texts, embedding_model):
    return embedding_model.encode(texts)

# Perform semantic search
def semantic_search(query, embeddings, texts, embedding_model, k=5):
    query_embedding = embedding_model.encode([query])
    similarities = np.dot(embeddings, query_embedding.T).squeeze()
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [texts[i] for i in top_k_indices]
