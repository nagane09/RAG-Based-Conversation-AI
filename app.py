import streamlit as st
from src.retrieve import retrieve_docs
from src.generate import generate_answer
from src.speech import text_to_speech

st.set_page_config(page_title="TinyLlama RAG AI")
st.title("Offline TinyLlama RAG AI")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Retrieving relevant knowledge..."):
        docs = retrieve_docs(query)

    with st.spinner("Generating answer..."):
        answer = generate_answer(query, docs)

    st.subheader("Answer")
    st.write(answer)

    audio_file = text_to_speech(answer)
    st.audio(audio_file)
