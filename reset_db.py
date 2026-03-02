import sqlite3
import sqlite_vec

DB_PATH = '/home/eiddra/mcp-servers/open-memory/data/memory.db'

def reset():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cursor = conn.cursor()
    
    tables = ["memories", "vectors", "waypoints", "vec_memories", "fts_memories"]
    for t in tables:
        print(f"Dropping table {t}...")
        cursor.execute(f"DROP TABLE IF EXISTS {t}")
    
    conn.commit()
    conn.close()
    print("Database reset complete.")

if __name__ == "__main__":
    reset()
