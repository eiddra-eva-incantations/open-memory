import sqlite3
conn = sqlite3.connect(":memory:", isolation_level=None)
conn.execute("CREATE TABLE foo (bar TEXT)")
conn.execute("INSERT INTO foo VALUES ('baz')")
try:
    conn.commit()
    print("Commit succeeds without issue.")
except Exception as e:
    print(f"Commit fails: {e}")
