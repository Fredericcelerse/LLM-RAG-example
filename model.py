from unsloth import FastLanguageModel
import torch

# Load the model with Hugging Face token
def load_model(hf_token, model_name="unsloth/mistral-7b-v0.3"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=5000,
        dtype=None,
        load_in_4bit=True,
        token=hf_token,
    )
    return model, tokenizer

# Generate a response based on the provided context
def generate_answer(query, context, model, tokenizer):
    prompt = f"""Based on the following information:

{context}

Please provide a concise and informative answer to the following question:
{query}

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
