from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model ON CPU (NO device_map)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32
)

# Explicitly move to CPU
model.to("cpu")

# Create pipeline (force CPU with device=-1)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

def build_prompt(context, question):
    return f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

def generate_answer(query, docs, max_new_tokens=128, top_p=0.9, temperature=0.7):
    docs = docs[:3]
    prompt = build_prompt("\n\n".join(docs), query)

    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature
    )
    return output[0]["generated_text"]
