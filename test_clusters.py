import sqlite3
import sqlite_vec
import json

DB_PATH = '/home/eiddra/mcp-servers/open-memory/data/memory.db'

def test_thresholds():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, content, mean_vec FROM memories")
    memories = cursor.fetchall()
    print(f"Total memories: {len(memories)}")
    
    for threshold in [0.9, 0.85, 0.8, 0.75, 0.7]:
        clusters = []
        visited = set()
        for mem_id, content, vec_blob in memories:
            if mem_id in visited: continue
            cursor.execute("""
                SELECT v.id, v.distance 
                FROM vectors v
                WHERE v.id != ? AND v.embedding MATCH ? AND k = 5
            """, (mem_id, vec_blob))
            
            cluster = [mem_id]
            rows = cursor.fetchall()
            if threshold == 0.7:
                print(f"Top distances for {mem_id}: {[r[1] for r in rows]}")
            
            for neighbor_id, dist in rows:
                sim = 1.0 - dist
                if sim >= threshold and neighbor_id not in visited:
                    cluster.append(neighbor_id)
                    visited.add(neighbor_id)
            if len(cluster) > 1:
                clusters.append(cluster)
                visited.add(mem_id)
        print(f"Threshold {threshold}: Found {len(clusters)} clusters.")

if __name__ == "__main__":
    test_thresholds()
