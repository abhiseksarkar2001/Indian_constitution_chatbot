import os
import faiss
import pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# === Configuration ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "/home/abhisek/Project/constitution_of_india_chat/faiss_index"
FAISS_META_PATH = "/home/abhisek/Project/constitution_of_india_chat/faiss_index_meta.pkl"
GROQ_API_KEY = "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"

# === Load models and index ===
@st.cache_resource(show_spinner=False)
def load_resources():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return model, index, metadata

embedding_model, faiss_index, metadata_list = load_resources()
client = Groq(api_key=GROQ_API_KEY)

# === Embedding & Search ===
def embed_text(text):
    return embedding_model.encode([text], normalize_embeddings=True)

def search_faiss(query, top_k=5):
    query_embedding = embed_text(query)
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [metadata_list[i] for i in indices[0] if i < len(metadata_list)]

# === Prompt Construction ===
def build_prompt(contexts, user_query):
    context_text = "\n\n".join(
        f"Article {ctx.get('article', 'N/A')}: {ctx.get('title', '')}\n{ctx.get('description', '')}"
        for ctx in contexts
    )
    return f"""You are a legal expert AI strictly specialized in the Constitution of India.

Instructions:
- Use *only* the articles provided below to answer.
- Do NOT invent or refer to any article not shown.
- Be concise and formal like a constitutional legal expert.
- Always cite the article number and title when giving an answer.
- If no article is relevant, reply: "No relevant article found in the Constitution for your query."

Retrieved Constitution Articles:
{context_text}

User Query:
{user_query}

Now respond using only the articles provided above.
"""

# === Groq LLM call ===
def call_groq(prompt, model="deepseek-r1-distill-llama-70b"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a legal assistant specializing in the Constitution of India."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# === Streamlit App ===
def main():
    st.set_page_config(page_title="Chat with Indian Constitution", page_icon="📜")
    st.title("📜 Chat with the Indian Constitution")
    st.markdown("Ask your legal questions based on constitutional rights and provisions.")

    user_query = st.text_input("👤 Enter your legal question:", placeholder="e.g., Can the government create a new state?")
    
    if st.button("🔍 Get Constitutional Answer"):
        if not user_query.strip():
            st.warning("Please enter a valid question.")
            return

        with st.spinner("🔎 Searching for relevant articles..."):
            matched_articles = search_faiss(user_query)

        if not matched_articles:
            st.error("No relevant article found in the Constitution for your query.")
            return

        st.subheader("📄 Top Matching Articles:")
        for i, art in enumerate(matched_articles, 1):
            st.markdown(f"**{i}. Article {art.get('article', 'N/A')} - {art.get('title', '')}**\n\n{art.get('description', '')}")

        prompt = build_prompt(matched_articles, user_query)

        with st.spinner("🧠 Generating expert legal response..."):
            response = call_groq(prompt)

        st.subheader("⚖️ Constitutional Answer:")
        st.success(response)

if __name__ == "__main__":
    main()
