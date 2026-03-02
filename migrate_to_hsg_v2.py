import json
import sqlite3
import uuid
import time
import sqlite_vec
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = '/home/eiddra/mcp-servers/open-memory/data/memory.db'
MODEL_NAME = 'Qwen/Qwen3-Embedding-4B'
VEC_DIM = 1024
device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True
}

print(f"Loading flagship embedder {MODEL_NAME} on {device} (4-bit)...")
# Load only the embedder for migration to save VRAM
model = SentenceTransformer(
    MODEL_NAME, 
    device=device, 
    trust_remote_code=True,
    model_kwargs={"quantization_config": bnb_config}
)

def encode_flagship(text):
    full_vec = model.encode(text).astype(np.float32)
    return full_vec[:VEC_DIM] if len(full_vec) > VEC_DIM else full_vec

def migrate():
    # Initialize schema without loading server-side models
    import mcp_server
    mcp_server.init_db()
    
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cursor = conn.cursor()

    # 2. Get legacy data from backup
    print("Fetching legacy data from memories_backup...")
    cursor.execute("SELECT id, content, primary_sector, created_at FROM memories_backup")
    legacy_rows = cursor.fetchall()
    print(f"Found {len(legacy_rows)} memories to migrate.")

    for row_id, content, old_sector, created_at in legacy_rows:
        print(f"Migrating memory {row_id}...")
        
        # Sector mapping
        primary = old_sector if old_sector in ["episodic", "semantic", "procedural", "emotional", "reflective"] else "semantic"
        
        # Embedding
        vec = encode_flagship(content)
        
        # New HSG Node
        new_id = str(uuid.uuid4())
        now = int(time.time())
        try:
            created_ts = int(time.mktime(time.strptime(created_at, '%Y-%m-%d %H:%M:%S')))
        except:
            created_ts = now

        cursor.execute("""
            INSERT INTO memories (id, content, primary_sector, tags, meta, created_at, updated_at, last_seen_at, salience, decay_lambda, mean_vec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            new_id, content, primary, json.dumps([]), json.dumps({"legacy_id": row_id}),
            created_ts, now, now, 1.0, 0.005,
            sqlite_vec.serialize_float32(vec.tolist())
        ))

        # Sector Vector
        cursor.execute(
            "INSERT INTO vectors(id, sector, embedding) VALUES (?, ?, ?)",
            (new_id, primary, sqlite_vec.serialize_float32(vec.tolist()))
        )
        print(f"Vector inserted for memory {row_id}")

    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM vectors")
    final_count = cursor.fetchone()[0]
    print(f"Migration finished. Total vectors in table: {final_count}")
    conn.close()
    print("Flagship HMD v2 migration complete!")

if __name__ == "__main__":
    migrate()
