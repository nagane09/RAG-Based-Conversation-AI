import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "data/processed_chunks/chunks.npy"
INDEX_FILE = "vector_store/index.faiss"

chunks = np.load(CHUNKS_FILE, allow_pickle=True)
index = faiss.read_index(INDEX_FILE)

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_docs(query, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]
