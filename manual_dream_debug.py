import json
from mcp_server import find_memory_clusters, synthesize_cluster, add_memory, get_db
import time

def manual_dream(threshold=0.60):
    print(f"Starting manual dream with threshold {threshold}...")
    clusters = find_memory_clusters(threshold=threshold)
    print(f"Found {len(clusters)} clusters.")
    
    if not clusters:
        print("No clusters found. Exiting.")
        return
    
    # Just synthesize the first cluster for testing
    cluster = clusters[0]
    print(f"Synthesizing cluster: {cluster}")
    res = synthesize_cluster(cluster)
    print(f"Synthesis result: {json.dumps(res, indent=2)}")
    
    if "error" in res:
        print("Synthesis failed.")
        return

    # Store report data
    report_data = {
        "insights": [res["truth"]],
        "conflicts": res.get("contradictions", []),
        "curiosities": res.get("curiosities", []),
        "ts": int(time.time())
    }
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO system_status (key, value, updated_at) VALUES (?, ?, ?)",
        ("last_dream_report", json.dumps(report_data), int(time.time()))
    )
    conn.commit()
    conn.close()
    print("Report data saved to system_status.")

if __name__ == "__main__":
    manual_dream(0.60)
