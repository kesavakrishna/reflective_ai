# memory.py

import os
import faiss
import numpy as np
import pandas as pd
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Parameters
EMBED_DIM = 768  # Gemini embedding size
TOP_K = 5
MEMORY_CSV_PATH = os.getenv('MEMORY_CSV_PATH', 'memories.csv')
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)

# Ensure storage file exists
if os.path.exists(MEMORY_CSV_PATH):
    df = pd.read_csv(MEMORY_CSV_PATH)
else:
    df = pd.DataFrame(columns=["id", "type", "text", "timestamp"])
    df.to_csv(MEMORY_CSV_PATH, index=False)

# FAISS index (FlatL2)
index = faiss.IndexFlatL2(EMBED_DIM)
# A list to hold text entries in the order they were added
memory_texts = df["text"].tolist()

# If there are existing rows, load their embeddings
if not df.empty:
    embeddings = []
    for txt in memory_texts:
        emb = genai.embed_content(
            model="models/embedding-001",
            content=txt,
            task_type="retrieval_document"
        )["embedding"]
        embeddings.append(np.array(emb, dtype="float32"))
    if embeddings:
        index.add(np.stack(embeddings, axis=0))

def embed_text(text: str) -> np.ndarray:
    """Use Gemini to embed the text into a vector."""
    resp = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    vec = np.array(resp["embedding"], dtype="float32")
    return vec

def store_memory(text: str, mem_type: str = "interaction"):
    """Add a new memory."""
    global df, memory_texts, index

    # Create new ID
    new_id = int(df["id"].max()) + 1 if not df.empty else 1
    ts = datetime.utcnow().isoformat()

    # Append to DataFrame
    new_row = {"id": new_id, "type": mem_type, "text": text, "timestamp": ts}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(MEMORY_CSV_PATH, index=False)

    # Embed and add to FAISS
    vec = embed_text(text)
    index.add(np.expand_dims(vec, 0))
    memory_texts.append(text)

def retrieve_memories(query: str, top_k: int = TOP_K, mem_type: str | None = None) -> list[str]:
    """Return the top_k most similar past memories to the query."""
    if index.ntotal == 0:
        return []

    # Handle empty query to retrieve latest memories instead of similarity search
    if not query:
        if mem_type:
            mems = df[df['type'] == mem_type]
        else:
            mems = df
        
        # Sort by timestamp to get the most recent
        recent_mems = mems.sort_values(by='timestamp', ascending=False).head(top_k)
        return recent_mems['text'].tolist()

    q_vec = embed_text(query)
    
    if mem_type:
        # This is not the most efficient way, but it will work for now
        # A better approach would be to use metadata filtering in FAISS if available
        # or have separate indexes per memory type.
        
        # Search for more results to have a better chance of finding the desired type
        k_search = max(top_k, index.ntotal) if top_k > 0 else index.ntotal
        if k_search == 0:
             return []
        D, I = index.search(x=np.expand_dims(q_vec, 0), k=int(k_search))
        
        results = []
        for idx in I[0]:
            if 0 <= idx < len(memory_texts):
                # Assumes `df` is aligned with `memory_texts` and `index`
                if df.iloc[idx]['type'] == mem_type:
                    results.append(memory_texts[idx])
                if len(results) == top_k:
                    break
        return results
    else:
        k_search = min(top_k, index.ntotal)
        if k_search == 0:
             return []
        D, I = index.search(x=np.expand_dims(q_vec, 0), k=int(k_search))
        results = []
        for idx in I[0]:
            if 0 <= idx < len(memory_texts):
                results.append(memory_texts[idx])
        return results
