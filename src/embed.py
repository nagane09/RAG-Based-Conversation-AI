import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

CHUNKS_FILE = "data/processed_chunks/chunks.npy"
INDEX_FILE = "vector_store/index.faiss"

chunks = np.load(CHUNKS_FILE, allow_pickle=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

if not os.path.exists("vector_store"):
    os.makedirs("vector_store")

faiss.write_index(index, INDEX_FILE)
np.save("vector_store/embeddings.npy", embeddings)

print("FAISS index created and saved")
