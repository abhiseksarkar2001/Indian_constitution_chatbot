import os
import datetime
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# === Configuration ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "/home/abhisek/Project/constitution_of_india_chat/faiss_index"
FAISS_META_PATH = "/home/abhisek/Project/constitution_of_india_chat/faiss_index_meta.pkl"
GROQ_API_KEY = "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"

# === Groq Client ===
client = Groq(api_key=GROQ_API_KEY)

# === Load embedding model ===
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === Load FAISS index and metadata ===
def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

index, metadata_list = load_faiss_index()

# === Embedding function ===
def embed_text(text):
    return embedding_model.encode([text], normalize_embeddings=True)

# === Search function ===
def search_faiss(query, top_k=5):
    query_embedding = embed_text(query)
    distances, indices = index.search(query_embedding, top_k)
    return [metadata_list[i] for i in indices[0] if i < len(metadata_list)]

# === Prompt construction ===
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

# === Interactive CLI ===
def main():
    print("📜 Chat with the Indian Constitution")
    print("Ask your legal questions based on constitutional rights and provisions.")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("👤 You: ").strip()
        if user_query.lower() == "exit":
            print("👋 Exiting. Stay constitutionally informed!")
            break

        print("\n🔍 Searching for relevant constitutional articles...")
        matched_articles = search_faiss(user_query)

        if not matched_articles:
            print("No relevant articles found.")
            continue

        print("\n📄 Top Matching Articles:")
        for i, art in enumerate(matched_articles, 1):
            print(f"{i}. Article {art.get('article', 'N/A')} - {art.get('title', '')}")

        full_prompt = build_prompt(matched_articles, user_query)

        print("\n🧠 Generating legal response...\n")
        response = call_groq(full_prompt)

        print("⚖️ Constitutional Answer:")
        print(response)
        print("\n" + "=" * 60 + "\n")

        # === Save Output ===
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "constitution_outputs"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"constitution_response_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("👤 User Query:\n")
            f.write(user_query + "\n\n")

            f.write("📄 Retrieved Articles:\n")
            for i, art in enumerate(matched_articles, 1):
                article = art.get("article", "N/A")
                title = art.get("title", "Untitled")
                desc = art.get("description", "No description")
                f.write(f"{i}. Article {article} - {title}\n{desc}\n\n")

            f.write("⚖️ Constitutional Answer:\n")
            f.write(response + "\n")

        print(f"📁 Answer saved to: {filepath}\n")

if __name__ == "__main__":
    main()
