import json
import sqlite3
import sys
import os
import time
import uuid
import re
import traceback
import sqlite_vec
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import BitsAndBytesConfig

# --- Configuration & Models ---
DB_PATH = '/home/eiddra/mcp-servers/open-memory/data/memory.db'
LOG_PATH = '/tmp/open-memory.log'
EMBED_MODEL_NAME = 'Qwen/Qwen3-Embedding-4B'
RERANK_MODEL_NAME = 'Qwen/Qwen3-Reranker-4B'
VEC_DIM = 1024 

# Detect CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
reranker = None

def get_models():
    """Lazy-load models to save VRAM when not in use."""
    global model, reranker
    if model is None or reranker is None:
        log(f"Lazy-loading flagship models on {device} (4-bit)...")
        bnb_config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True
        }
        if model is None:
            model = SentenceTransformer(
                EMBED_MODEL_NAME, 
                device=device, 
                trust_remote_code=True,
                model_kwargs={"quantization_config": bnb_config}
            )
        if reranker is None:
            reranker = CrossEncoder(
                RERANK_MODEL_NAME, 
                device=device, 
                trust_remote_code=True,
                model_kwargs={"quantization_config": bnb_config}
            )
            if reranker.tokenizer.pad_token is None:
                reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
                reranker.model.config.pad_token_id = reranker.tokenizer.pad_token_id
    return model, reranker

def log(msg: str):
    with open(LOG_PATH, 'a') as f:
        f.write(f"[{time.ctime()}] {msg}\n")
        f.flush()

# --- HMD v2 Constants ---
SECTORS = {
    "episodic": {"decay_lambda": 0.015, "weight": 1.2, "patterns": [r"today", r"yesterday", r"remember when", r"happened"]},
    "semantic": {"decay_lambda": 0.005, "weight": 1.0, "patterns": [r"define", r"meaning", r"concept", r"is a", r"are"]},
    "procedural": {"decay_lambda": 0.008, "weight": 1.1, "patterns": [r"how to", r"step by step", r"workflow", r"process"]},
    "emotional": {"decay_lambda": 0.020, "weight": 1.3, "patterns": [r"feel", r"happy", r"sad", r"angry", r"preference"]},
    "reflective": {"decay_lambda": 0.001, "weight": 0.8, "patterns": [r"think", r"realize", r"insight", r"goal", r"philosophy"]}
}

# --- Database Layer ---
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # 1. Memories Table (HSG Nodes)
    cursor.execute("CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY)")
    
    # Add columns if they don't exist (Migration logic)
    cols = [
        ("content", "TEXT NOT NULL DEFAULT ''"),
        ("primary_sector", "TEXT NOT NULL DEFAULT 'semantic'"),
        ("tags", "TEXT"),
        ("meta", "TEXT"),
        ("created_at", "INTEGER"),
        ("updated_at", "INTEGER"),
        ("last_seen_at", "INTEGER"),
        ("salience", "REAL DEFAULT 1.0"),
        ("decay_lambda", "REAL DEFAULT 0.005"),
        ("version", "INTEGER DEFAULT 2"),
        ("mean_vec", "BLOB")
    ]
    for col_name, col_type in cols:
        try:
            cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass # Already exists
    
    # 2. Vector Table
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
            id TEXT,
            sector TEXT,
            embedding float[1024]
        )
    """)
    
    # 3. Waypoint Graph
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS waypoints (
            src_id TEXT PRIMARY KEY,
            dst_id TEXT NOT NULL,
            weight REAL NOT NULL,
            created_at INTEGER,
            updated_at INTEGER
        )
    """)
    
    # 4. Temporal Facts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS temporal_facts (
            id TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            content TEXT,
            confidence REAL DEFAULT 1.0,
            valid_from INTEGER,
            valid_to INTEGER,
            created_at INTEGER
        )
    """)
    
    # 5. FTS5
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
            content,
            content='memories',
            content_rowid='id'
        )
    """)
    
    # 6. System Status Table (for Dreaming Loop)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_status (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at INTEGER
        )
    """)
    
    conn.commit()
    conn.close()

# --- Cognitive Logic ---

def classify_content(content: str) -> Tuple[str, List[str]]:
    scores = {s: 0 for s in SECTORS}
    for sector, config in SECTORS.items():
        for pattern in config["patterns"]:
            if re.search(pattern, content, re.IGNORECASE):
                scores[sector] += 1
    sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_sectors[0][0] if sorted_sectors[0][1] > 0 else "semantic"
    additional = [s for s, score in sorted_sectors[1:] if score > 0]
    return primary, additional

def calculate_decay(initial_salience: float, decay_lambda: float, last_seen_ts: int) -> float:
    days = (time.time() - last_seen_ts) / (24 * 3600)
    return initial_salience * np.exp(-decay_lambda * days)

def get_mean_vector(vectors: List[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros(VEC_DIM)
    return np.mean(vectors, axis=0)

def encode_flagship(text: str) -> np.ndarray:
    """Encode text and trim to VEC_DIM (Matryoshka)."""
    model, _ = get_models()
    full_vec = model.encode(text).astype(np.float32)
    return full_vec[:VEC_DIM] if len(full_vec) > VEC_DIM else full_vec

# --- Dreaming & Synthesis Logic ---

def find_memory_clusters(threshold: float = 0.85) -> List[List[str]]:
    """Find clusters of memories with high cosine similarity."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        # Fetch all memories with mean vectors
        cursor.execute("SELECT id, mean_vec FROM memories WHERE primary_sector != 'reflective'")
        all_memories = cursor.fetchall()
        
        clusters = []
        visited = set()
        
        for mem_id, vec_blob in all_memories:
            if mem_id in visited: continue
            
            # Find neighbors for this memory in the VIRTUAL vectors table
            cursor.execute("""
                SELECT v.id, (1.0 - v.distance) as similarity 
                FROM vectors v 
                WHERE v.id != ? AND v.embedding MATCH ? AND k = 10
            """, (mem_id, vec_blob))
            
            cluster = [mem_id]
            for neighbor_id, sim in cursor.fetchall():
                if sim >= threshold and neighbor_id not in visited:
                    cluster.append(neighbor_id)
                    visited.add(neighbor_id)
            
            if len(cluster) > 1:
                clusters.append(cluster)
                visited.add(mem_id)
                
        return clusters
    finally:
        conn.close()

def dream(threshold: float = 0.60):
    """Execute the Clustering phase and return clusters for Gemini to synthesize."""
    clusters = find_memory_clusters(threshold=threshold)
    log(f"Dreaming: Found {len(clusters)} memory clusters at threshold {threshold}.")
    
    conn = get_db()
    cursor = conn.cursor()
    cluster_data = []
    
    try:
        for cluster_ids in clusters:
            placeholders = ','.join(['?'] * len(cluster_ids))
            cursor.execute(f"SELECT id, content, primary_sector, meta FROM memories WHERE id IN ({placeholders})", cluster_ids)
            rows = cursor.fetchall()
            cluster_data.append([
                {"id": r[0], "content": r[1], "sector": r[2], "meta": json.loads(r[3])}
                for r in rows
            ])
        return cluster_data
    finally:
        conn.close()

def generate_morning_report(data: Dict):
    """Store the final synthesized report data for easy retrieval."""
    log(f"Archiving Morning Report with {len(data.get('insights', []))} insights.")
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO system_status (key, value, updated_at) VALUES (?, ?, ?)",
            ("last_dream_report", json.dumps(data), int(time.time()))
        )
        conn.commit()
        return "Report data archived in system_status."
    except Exception as e:
        log(f"Error archiving report: {e}")
        return f"Error: {e}"
    finally:
        conn.close()

def ingest_eiddra_replies(email_body: str):
    """Pass-through for Eiddra's feedback to be processed by Gemini."""
    # Store the raw feedback for reference
    add_memory(
        content=f"Eiddra's Raw Feedback: {email_body}",
        type="reflective",
        tags=["feedback", "raw", "dreaming-loop"]
    )
    return "Feedback stashed in reflective memory. Ready for synthesis."

def create_github_issue(title: str, body: str, labels: List[str] = None):
    """Create a GitHub Issue using the CLI."""
    import subprocess
    cmd = ["gh", "issue", "create", "--title", title, "--body", body]
    if labels:
        for label in labels:
            cmd.extend(["--label", label])
    
    # Run from the repo directory
    result = subprocess.run(cmd, cwd="/home/eiddra/mcp-servers/open-memory", capture_output=True, text=True)
    if result.returncode == 0:
        return f"Issue created: {result.stdout.strip()}"
    else:
        return f"Error creating issue: {result.stderr}"

def fetch_github_comments(issue_number: int):
    """Fetch comments for a specific issue using the CLI."""
    import subprocess
    cmd = ["gh", "issue", "view", str(issue_number), "--comments", "--json", "comments"]
    result = subprocess.run(cmd, cwd="/home/eiddra/mcp-servers/open-memory", capture_output=True, text=True)
    if result.returncode == 0:
        data = json.loads(result.stdout)
        return data.get("comments", [])
    else:
        return f"Error fetching comments: {result.stderr}"

def backup_to_github(message: str = None):
    """Perform a git-based backup of the server and database."""
    import subprocess
    if message is None:
        message = f"Nightly Backup - {time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    try:
        # Add files
        subprocess.run(["git", "add", "."], cwd="/home/eiddra/mcp-servers/open-memory", check=True)
        # Commit
        subprocess.run(["git", "commit", "-m", message], cwd="/home/eiddra/mcp-servers/open-memory", check=True)
        # Push
        subprocess.run(["git", "push", "origin", "master"], cwd="/home/eiddra/mcp-servers/open-memory", check=True)
        return "Backup successful and pushed to GitHub."
    except Exception as e:
        return f"Backup failed: {e}"

# --- Core Operations ---

def add_memory(content: str, type: str = "contextual", facts: List[Dict] = None, tags: List[str] = None, metadata: Dict = None) -> Dict:
    model, _ = get_models()
    conn = get_db()
    cursor = conn.cursor()
    now = int(time.time())
    mem_id = str(uuid.uuid4())
    
    try:
        results = {"id": mem_id}
        if type in ["contextual", "both"]:
            primary, additional = classify_content(content)
            all_sectors = [primary] + additional
            sector_vectors = {}
            for sector in all_sectors:
                vec = encode_flagship(content)
                sector_vectors[sector] = vec
            mean_vec = get_mean_vector(list(sector_vectors.values()))
            cursor.execute("""
                INSERT INTO memories (id, content, primary_sector, tags, meta, created_at, updated_at, last_seen_at, salience, decay_lambda, mean_vec)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mem_id, content, primary, json.dumps(tags or []), json.dumps(metadata or {}),
                now, now, now, 1.0, SECTORS[primary]["decay_lambda"], 
                sqlite_vec.serialize_float32(mean_vec.tolist())
            ))
            for sector, vec in sector_vectors.items():
                cursor.execute(
                    "INSERT INTO vectors(id, sector, embedding) VALUES (?, ?, ?)",
                    (mem_id, sector, sqlite_vec.serialize_float32(vec.tolist()))
                )
            cursor.execute("""
                SELECT id, (1.0 - distance) as similarity 
                FROM memories 
                WHERE id != ? AND mean_vec MATCH ? AND k = 1
            """, (mem_id, sqlite_vec.serialize_float32(mean_vec.tolist())))
            match = cursor.fetchone()
            if match and match[1] >= 0.75:
                cursor.execute("""
                    INSERT OR REPLACE INTO waypoints (src_id, dst_id, weight, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (mem_id, match[0], float(match[1]), now, now))
            results["hsg"] = {"primary_sector": primary, "sectors": all_sectors}

        if type in ["factual", "both"] and facts:
            temporal_results = []
            for fact in facts:
                f_id = str(uuid.uuid4())
                subject = fact["subject"]
                predicate = fact["predicate"]
                obj = fact["object"]
                cursor.execute("""
                    UPDATE temporal_facts 
                    SET valid_to = ? 
                    WHERE subject = ? AND predicate = ? AND valid_to IS NULL
                """, (now, subject, predicate))
                cursor.execute("""
                    INSERT INTO temporal_facts (id, subject, predicate, object, content, confidence, valid_from, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (f_id, subject, predicate, obj, content, fact.get("confidence", 1.0), fact.get("valid_from", now), now))
                temporal_results.append({"id": f_id, "subject": subject, "predicate": predicate, "object": obj})
            results["temporal"] = temporal_results
        conn.commit()
        return results
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def query_memories(query: str, type: str = "contextual", sector: str = None, k: int = 8, at: int = None) -> Dict:
    model, reranker = get_models()
    conn = get_db()
    cursor = conn.cursor()
    now = int(time.time())
    try:
        results = {"type": type}
        if type in ["contextual", "unified"]:
            query_vec = encode_flagship(query)
            candidate_query = """
                SELECT m.id, m.content, m.primary_sector, m.salience, m.decay_lambda, m.last_seen_at, m.created_at, (1.0 - v.distance) as sim
                FROM memories m
                JOIN vectors v ON m.id = v.id
                WHERE v.embedding MATCH ? AND k = ?
            """
            params = [sqlite_vec.serialize_float32(query_vec.tolist()), k * 2]
            if sector:
                candidate_query = candidate_query.replace("WHERE", "WHERE v.sector = ? AND")
                params = [sector] + params
            cursor.execute(candidate_query, params)
            candidates = {}
            for row in cursor.fetchall():
                salience = calculate_decay(row[3], row[4], row[5])
                recency = 1.0 / (1.0 + (now - row[6]) / (3600 * 24 * 30))
                candidates[row[0]] = {
                    "id": row[0], "content": row[1], "primary_sector": row[2],
                    "sim": row[7], "salience": salience, "recency": recency, "waypoint": 0.0
                }
            found_ids = list(candidates.keys())
            if found_ids:
                placeholders = ','.join(['?'] * len(found_ids))
                cursor.execute(f"""
                    SELECT w.dst_id, w.weight, m.content, m.primary_sector, m.salience, m.decay_lambda, m.last_seen_at, m.created_at
                    FROM waypoints w
                    JOIN memories m ON w.dst_id = m.id
                    WHERE w.src_id IN ({placeholders})
                """, found_ids)
                for row in cursor.fetchall():
                    if row[0] not in candidates:
                        salience = calculate_decay(row[4], row[5], row[6])
                        recency = 1.0 / (1.0 + (now - row[7]) / (3600 * 24 * 30))
                        candidates[row[0]] = {
                            "id": row[0], "content": row[2], "primary_sector": row[3],
                            "sim": 0.0, "salience": salience, "recency": recency, "waypoint": row[1]
                        }
                    else:
                        candidates[row[0]]["waypoint"] = max(candidates[row[0]]["waypoint"], row[1])
            candidate_list = list(candidates.values())
            if candidate_list:
                pairs = [(query, c["content"]) for c in candidate_list]
                # Force batch_size=1 to avoid pad_token issues in Qwen3-Reranker
                rerank_scores = reranker.predict(pairs, batch_size=1)
                for i, score in enumerate(rerank_scores):
                    c = candidate_list[i]
                    c["score"] = (0.6 * float(score)) + (0.2 * c["salience"]) + (0.1 * c["recency"]) + (0.1 * c["waypoint"])
                candidate_list.sort(key=lambda x: x["score"], reverse=True)
                results["contextual"] = candidate_list[:k]
                if candidate_list:
                    top_id = candidate_list[0]["id"]
                    cursor.execute("UPDATE memories SET last_seen_at = ?, salience = MIN(1.0, salience + 0.1) WHERE id = ?", (now, top_id))
        if type in ["factual", "unified"]:
            target_time = at if at else now
            cursor.execute("""
                SELECT id, subject, predicate, object, confidence, valid_from, valid_to
                FROM temporal_facts
                WHERE (valid_to IS NULL OR valid_to > ?) AND valid_from <= ?
            """, (target_time, target_time))
            results["factual"] = [
                {"id": r[0], "subject": r[1], "predicate": r[2], "object": r[3], "confidence": r[4], "valid_from": r[5], "valid_to": r[6]}
                for r in cursor.fetchall()
            ]
        conn.commit()
        return results
    finally:
        conn.close()

def list_tools():
    return {
        "tools": [
            {
                "name": "openmemory_store",
                "description": "Store content in contextual memory (HSG), temporal facts, or both (1:1 HMD v2).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Raw memory text."},
                        "type": {"type": "string", "enum": ["contextual", "factual", "both"], "default": "contextual"},
                        "facts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "subject": {"type": "string"},
                                    "predicate": {"type": "string"},
                                    "object": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "valid_from": {"type": "integer"}
                                },
                                "required": ["subject", "predicate", "object"]
                            }
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "metadata": {"type": "object"}
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "openmemory_query",
                "description": "Query contextual memories, temporal facts, or both using HMD v2 flagship re-ranking.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "type": {"type": "string", "enum": ["contextual", "factual", "unified"], "default": "contextual"},
                        "k": {"type": "integer", "default": 8},
                        "sector": {"type": "string", "enum": ["episodic", "semantic", "procedural", "emotional", "reflective"]},
                        "at": {"type": "integer", "description": "Epoch timestamp for point-in-time factual queries."}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "openmemory_dream",
                "description": "Trigger the Reflective Synthesis (Dreaming) cycle to distill episodic logs into semantic wisdom.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "threshold": {"type": "number", "description": "Clustering similarity threshold (default 0.85)."}
                    }
                }
            },
            {
                "name": "openmemory_generate_morning_report",
                "description": "Generate and send the Morning Cognitive Report to Eiddra via Gmail.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "description": "Target email address (default eiddra@proton.me)."},
                        "data": {"type": "object"}
                    }
                }
            },
            {
                "name": "openmemory_ingest_replies",
                "description": "Search for and ingest Eiddra's email replies to update synthesized facts.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "email_body": {"type": "string", "description": "The content of Eiddra's reply email."}
                    },
                    "required": ["email_body"]
                }
            },
            {
                "name": "openmemory_create_issue",
                "description": "Create a GitHub Issue for a cognitive sticking point or curiosity.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "labels": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["title", "body"]
                }
            },
            {
                "name": "openmemory_fetch_comments",
                "description": "Fetch comments from GitHub Issues to ingest feedback.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "issue_number": {"type": "integer"}
                    },
                    "required": ["issue_number"]
                }
            },
            {
                "name": "openmemory_backup",
                "description": "Perform a nightly backup of the cognitive setup and database to GitHub.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Commit message."}
                    }
                }
            }
        ]
    }

def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    method = request.get("method")
    params = request.get("params", {})
    if method == "initialize":
        init_db()
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "open-memory-flagship", "version": "2.0.0"}
        }
    elif method == "notifications/initialized": return None
    elif method == "tools/list": return list_tools()
    elif method == "tools/call" or method == "callTool":
        tool_name = params.get("name") or params.get("tool")
        args = params.get("arguments", {})
        try:
            if tool_name == "openmemory_store":
                result = add_memory(args["content"], args.get("type", "contextual"), args.get("facts"), args.get("tags"), args.get("metadata"))
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            elif tool_name == "openmemory_query":
                result = query_memories(args["query"], args.get("type", "contextual"), args.get("sector"), args.get("k", 8), args.get("at"))
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            elif tool_name == "openmemory_dream":
                result = dream(args.get("threshold", 0.60))
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            elif tool_name == "openmemory_generate_morning_report":
                result = generate_morning_report(args["data"])
                return {"content": [{"type": "text", "text": result}]}
            elif tool_name == "openmemory_ingest_replies":
                result = ingest_eiddra_replies(args["email_body"])
                return {"content": [{"type": "text", "text": result}]}
            elif tool_name == "openmemory_create_issue":
                result = create_github_issue(args["title"], args["body"], args.get("labels"))
                return {"content": [{"type": "text", "text": result}]}
            elif tool_name == "openmemory_fetch_comments":
                result = fetch_github_comments(args["issue_number"])
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            elif tool_name == "openmemory_backup":
                result = backup_to_github(args.get("message"))
                return {"content": [{"type": "text", "text": result}]}
        except Exception as e:
            return {"error": {"code": -32603, "message": str(e), "data": traceback.format_exc()}}
    return {"error": {"code": -32601, "message": f"Method not found: {method}"}}

def main():
    log(f"Server starting (HMD v2 Flagship, 4-bit CUDA, Device: {device})...")
    for line in sys.stdin:
        if not line.strip(): continue
        try:
            request = json.loads(line)
            response = handle_request(request)
            if response is None: continue
            output = {"jsonrpc": "2.0", "id": request.get("id"), "result": response}
            print(json.dumps(output))
            sys.stdout.flush()
        except Exception as e:
            log(f"Fatal error in main loop: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
