# Offline TinyLlama RAG AI System

This project is an **offline Retrieval-Augmented Generation (RAG) AI system** built using **TinyLlama**, **FAISS**, and **Sentence Transformers**. It allows users to ask questions about local PDF documents and receive AI-generated answers **without internet access**, along with **text-to-speech output**.

---

## **Key Features**

* Fully **offline RAG system** (no API calls, no internet required)
* PDF document ingestion and chunking
* Semantic search using **FAISS vector store**
* Context-aware answer generation using **TinyLlama-1.1B**
* **Streamlit web interface** for interaction
* **Text-to-Speech (TTS)** output using `pyttsx3`

---

## **Technologies Used**

* Python
* FAISS (Vector similarity search)
* Sentence Transformers (`all-MiniLM-L6-v2`)
* TinyLlama-1.1B-Chat
* Hugging Face Transformers
* PyPDF
* Streamlit
* pyttsx3 (Offline Text-to-Speech)
* NumPy, Torch

---

## **How It Works**

1. **PDF Ingestion**
   * PDFs are read from `data/raw_docs/`
   * Text is extracted and split into overlapping chunks

2. **Vector Index Creation**
   * Each chunk is embedded using SentenceTransformer
   * FAISS index is created and stored locally

3. **Query Processing**
   * User query is embedded
   * FAISS retrieves top-k relevant chunks

4. **Answer Generation**
   * Retrieved chunks are injected into a prompt
   * TinyLlama generates a context-aware answer

5. **Speech Output**
   * Answer is converted into speech and played in the app

---

## **How to Run Locally**

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```
### **2. Add PDF Documents**

```bash
data/raw_docs/
```
### **3. Extract and Chunk PDFs**

```bash
python src/ingest.py
```
### **This will generate:--**

```bash
data/processed_chunks/chunks.npy
```
### **4. Build FAISS Vector Index**

```bash
python src/build_index.py
```
### **This will generate:--**

```bash
vector_store/index.faiss
vector_store/embeddings.npy
```
### **5. Run the Streamlit App**

```bash
streamlit run src/app.py
```

### **6. Ask Questions**

* Type your question in the input box.
* The system retrieves relevant document chunks using FAISS.
* TinyLlama generates a context-aware answer.
* The answer is also played as **audio output**.



