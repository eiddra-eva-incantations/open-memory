import sqlite3
import sqlite_vec

DB_PATH = '/home/eiddra/mcp-servers/open-memory/data/memory.db'

def check():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM memories")
    print(f"Memories count: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM vectors")
    print(f"Vectors count: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT id, sector FROM vectors LIMIT 5")
    print(f"Sample vectors: {cursor.fetchall()}")
    
    conn.close()

if __name__ == "__main__":
    check()
