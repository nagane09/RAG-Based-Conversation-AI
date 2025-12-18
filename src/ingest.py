from pypdf import PdfReader
import numpy as np
import os

RAW_DIR = "data/raw_docs"
OUT_FILE = "data/processed_chunks/chunks.npy"

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

all_chunks = []

for file in os.listdir(RAW_DIR):
    if file.endswith(".pdf"):
        text = load_pdf(os.path.join(RAW_DIR, file))
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

if not os.path.exists("data/processed_chunks"):
    os.makedirs("data/processed_chunks")

np.save(OUT_FILE, np.array(all_chunks, dtype=object))
print(f"Saved {len(all_chunks)} chunks")
