# ChatBot with RAG (Retrieval-Augmented Generation): a simple pipeline to start

This project implements a pipeline that combines semantic search with a pre-trained language model to answer questions based on relevant information extracted from web pages. The system leverages the retrieval-augmented generation (RAG) approach, where text embeddings are used for semantic search, and a language model generates human-readable answers.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Functions Documentation](#functions-documentation)
- [Parameters Explanation](#parameters-explanation)
- [Google Colab](#google-colab)

---

## Installation

To set up the environment locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/yourproject.git
    cd yourproject
    ```

2. **Create a Python virtual environment (optional but recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Provide your Hugging Face token**:
    - Update the Hugging Face token in `main.py` with your own token from Hugging Face.
    - You can get a token from your Hugging Face account [here](https://huggingface.co/settings/tokens).

---

## Usage

To run the project, use the following command:

```bash
python main.py
```

When you run the script, it will:

- Load a pre-trained language model from Hugging Face.
- Retrieve and extract relevant content from the given URLs.
- Chunk the text into smaller parts for embedding.
- Perform a semantic search using embeddings.
- Generate a human-readable answer to the query based on the most relevant chunks.
  
You can customize the `urls` and `query` directly in the `main.py` file.

---

## Functions Documentation

### `model.py`

#### `load_model(hf_token, model_name)`

- **Loads the pre-trained language model from Hugging Face using the provided token.**
  
  **Parameters**:
  - `hf_token`: Hugging Face API token.
  - `model_name`: The name of the pre-trained model (default: `"unsloth/mistral-7b-v0.3"`).
  
  **Returns**: The loaded model and tokenizer.

#### `generate_answer(query, context, model, tokenizer)`

- **Generates a natural language answer based on the input query and context.**
  
  **Parameters**:
  - `query`: The user’s question.
  - `context`: The text context from which the answer will be generated.
  - `model`: The pre-trained language model.
  - `tokenizer`: Tokenizer associated with the model.
  
  **Returns**: The generated answer as a string.

---

### `embeddings.py`

#### `get_embedding_model(model_name)`

- **Loads the selected embedding model for generating text embeddings.**
  
  **Parameters**:
  - `model_name`: The name of the embedding model (e.g., `"jina-embeddings-v2"`, `"longformer"`, `"e5-large"`).
  
  **Returns**: The loaded embedding model.

#### `get_embeddings(texts, embedding_model)`

- **Generates text embeddings for a list of texts.**
  
  **Parameters**:
  - `texts`: A list of texts to be embedded.
  - `embedding_model`: The embedding model to be used.
  
  **Returns**: A list of embeddings for the provided texts.

#### `semantic_search(query, embeddings, texts, embedding_model, k=5)`

- **Performs semantic search to retrieve the top `k` relevant texts based on a query.**
  
  **Parameters**:
  - `query`: The query in natural language.
  - `embeddings`: Pre-computed embeddings for the text chunks.
  - `texts`: The text chunks corresponding to the embeddings.
  - `embedding_model`: The embedding model used to compute the embeddings.
  - `k`: The number of top results to retrieve (default: `5`).
  
  **Returns**: A list of the top `k` relevant texts.

---

### `utils.py`

#### `extract_content(url)`

- **Extracts the main content from a web page.**
  
  **Parameters**:
  - `url`: The URL of the web page.
  
  **Returns**: The extracted text content as a string.

#### `chunk_text(text, chunk_size=200)`

- **Splits a large text into smaller chunks of a specified size.**
  
  **Parameters**:
  - `text`: The full text to be chunked.
  - `chunk_size`: The size of each chunk in terms of characters (default: `200`).
  
  **Returns**: A list of text chunks.

---

### `main.py`

#### `rag_answer(query, urls, hf_token)`

- **Combines retrieval and generation steps to provide an answer based on relevant web page content.**
  
  **Parameters**:
  - `query`: The user’s question in natural language.
  - `urls`: A list of URLs from which relevant information will be extracted.
  - `hf_token`: The Hugging Face token to authenticate the model.
  
  **Returns**: A generated answer in response to the query.

---

## Parameters Explanation

#### `hf_token`

- Hugging Face token required to authenticate and load the pre-trained model. It is necessary to access models from the Hugging Face Hub.

#### `model_name`

- The name of the pre-trained language model (e.g., `"unsloth/mistral-7b-v0.3"`). Different models might have varying performance depending on the task.

#### `embedding_model`

- The embedding model used to generate vector representations of text chunks. The model chosen here influences the quality and accuracy of the semantic search.

#### `chunk_size`

- Determines how the text is split into smaller parts. Smaller chunks may lead to more precise retrieval but may miss some larger contextual information.

#### `query`

- The natural language question or query you are asking. The quality of the query influences the final answer generated by the model.

#### `urls`

- List of URLs that the system will extract content from. These URLs should contain information relevant to the query.

#### `k` (in `semantic_search`)

- The number of top results to retrieve during the semantic search. Increasing `k` will return more relevant chunks but may also include less pertinent ones.

---

This README provides detailed instructions on how to use and understand the code, along with documentation of the main functions and parameters. You can modify the parameters according to your needs to fine-tune the results.
