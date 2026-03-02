import json
import sqlite3
import sqlite_vec
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

DB_PATH = '/home/eiddra/mcp-servers/open-memory/data/memory.db'
EMBED_MODEL_NAME = 'Qwen/Qwen3-Embedding-4B'
RERANK_MODEL_NAME = 'Qwen/Qwen3-Reranker-4B'
VEC_DIM = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"
bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True
}

print(f"Loading flagship models on {device} (4-bit)...")
model = SentenceTransformer(EMBED_MODEL_NAME, device=device, trust_remote_code=True, model_kwargs={"quantization_config": bnb_config})
reranker = CrossEncoder(RERANK_MODEL_NAME, device=device, trust_remote_code=True, model_kwargs={"quantization_config": bnb_config})

if reranker.tokenizer.pad_token is None:
    reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
    reranker.model.config.pad_token_id = reranker.tokenizer.pad_token_id

def debug_query(query):
    print(f"Debugging query: '{query}'")
    full_vec = model.encode(query).astype(np.float32)
    query_vec = full_vec[:VEC_DIM] if len(full_vec) > VEC_DIM else full_vec
    
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cursor = conn.cursor()
    
    print("Checking vectors table count...")
    cursor.execute("SELECT COUNT(*) FROM vectors")
    print(f"Vectors in table: {cursor.fetchone()[0]}")
    
    # Vector Search
    print("Performing vector search...")
    cursor.execute("""
        SELECT m.id, m.content, (1.0 - v.distance) as sim
        FROM memories m
        JOIN vectors v ON m.id = v.id
        WHERE v.embedding MATCH ? AND k = 10
    """, (sqlite_vec.serialize_float32(query_vec.tolist()),))
    candidates = cursor.fetchall()
    print(f"Found {len(candidates)} candidates.")
    for c in candidates:
        print(f"ID: {c[0]}, Sim: {c[2]:.4f}, Content: {c[1][:50]}...")

    if candidates:
        pairs = [(query, c[1]) for c in candidates]
        scores = reranker.predict(pairs, batch_size=1)
        print("Re-ranked scores top 3:")
        results = [{"content": c[1], "score": float(s)} for c, s in zip(candidates, scores)]
        results.sort(key=lambda x: x["score"], reverse=True)
        print(json.dumps(results[:3], indent=2))

if __name__ == "__main__":
    debug_query("EleusinianSuite")
