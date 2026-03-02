import sqlite3
import sqlite_vec
import numpy as np

DB_PATH = '/home/eiddra/mcp-servers/open-memory/data/memory.db'

def debug():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cursor = conn.cursor()
    
    print("Fetching a sample memory and its mean_vec...")
    cursor.execute("SELECT id, content, mean_vec FROM memories LIMIT 1")
    row = cursor.fetchone()
    if not row:
        print("No memories found.")
        return
    
    mem_id, content, vec_blob = row
    print(f"Memory ID: {mem_id}")
    
    # Try MATCH search
    print("Performing MATCH search with sample vec_blob...")
    cursor.execute("""
        SELECT v.id, v.distance 
        FROM vectors v 
        WHERE v.embedding MATCH ? AND k = 5
    """, (vec_blob,))
    rows = cursor.fetchall()
    print(f"MATCH results: {rows}")
    
    # Try direct distance calculation
    print("Performing manual distance check...")
    cursor.execute("SELECT id, embedding FROM vectors LIMIT 5")
    vecs = cursor.fetchall()
    for vid, v_emb in vecs:
        # sqlite-vec has vec_distance_cosine
        cursor.execute("SELECT vec_distance_cosine(?, ?)", (vec_blob, v_emb))
        dist = cursor.fetchone()[0]
        print(f"ID: {vid}, Cosine Dist: {dist:.4f}")

    conn.close()

if __name__ == "__main__":
    debug()
