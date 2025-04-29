import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# === Configuration ===
INPUT_JSON_PATH = '/home/abhisek/Project/constitution_of_india_chat/constitution_of_india.json'
OUTPUT_DIR = '/home/abhisek/Project/constitution_of_india_chat'

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'faiss_index')
FAISS_META_PATH = os.path.join(OUTPUT_DIR, 'faiss_index_meta.pkl')

# === Create Output Directory if not exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Embedding Model ===
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === Normalize and extract text for embedding ===
def normalize_json(json_obj):
    items = []
    for entry in json_obj:
        try:
            article = entry.get('article', 'N/A')
            title = entry.get('title', 'Untitled')
            desc = entry.get('description', '')

            # Compose final text for embedding
            content = f"Article {article}: {title}\n{desc}".strip()

            items.append({
                'article': article,
                'title': title,
                'description': desc,
                'text': content
            })
        except Exception as e:
            print(f"❌ Error parsing entry: {e}")
    return items

# === Load and process JSON ===
all_texts = []
metadata = []

try:
    with open(INPUT_JSON_PATH, 'r') as f:
        raw_data = json.load(f)
        extracted = normalize_json(raw_data)
        for item in extracted:
            all_texts.append(item['text'])
            metadata.append(item)
except Exception as e:
    print(f"❌ Error loading {INPUT_JSON_PATH}: {e}")

print(f"✅ Total articles loaded for embedding: {len(all_texts)}")

# === Generate Embeddings ===
embeddings = model.encode(all_texts, normalize_embeddings=True)
embeddings = np.array(embeddings).astype('float32')

# === Create and save FAISS index ===
index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product (cosine similarity)
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)
with open(FAISS_META_PATH, 'wb') as f:
    pickle.dump(metadata, f)

print("✅ Embedding & Indexing completed.")
print(f"📁 FAISS Index saved to: {FAISS_INDEX_PATH}")
print(f"📁 Metadata saved to: {FAISS_META_PATH}")
