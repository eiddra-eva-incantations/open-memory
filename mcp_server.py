import json
import sqlite3
import sys
import os
import time
import uuid
import re
import traceback
import subprocess
import threading
import fcntl
import gc
import sqlite_vec
import urllib.request
import urllib.error
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration & Models ---
DB_PATH = '/home/eiddra/mcp-servers/open-memory/data/memory.db'
LOG_PATH = '/tmp/open-memory.log'
EMBED_MODEL_NAME = 'Qwen/Qwen3-Embedding-4B'
RERANK_MODEL_NAME = 'Qwen/Qwen3-Reranker-4B'
VEC_DIM = 1024 


# Daemon config
MODEL_DAEMON_URL = "http://127.0.0.1:50051"


LOG_LEVEL = os.environ.get("OPENMEMORY_LOG_LEVEL", "INFO").upper()
LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}

def log(msg: str, level: str = "INFO"):
    if LOG_LEVELS.get(level, 1) < LOG_LEVELS.get(LOG_LEVEL, 1):
        return
    # Simple log rotation: truncate if > 5 MB
    try:
        if os.path.getsize(LOG_PATH) > 5 * 1024 * 1024:
            with open(LOG_PATH, 'r') as f:
                lines = f.readlines()
            with open(LOG_PATH, 'w') as f:
                f.writelines(lines[-1000:])
    except (OSError, FileNotFoundError):
        pass
    with open(LOG_PATH, 'a') as f:
        log_line = f"[{time.ctime()}] [{level}] {msg}\n"
        f.write(log_line)
        f.flush()
    # Also print to stderr for terminal visibility
    sys.stderr.write(log_line)
    sys.stderr.flush()

# --- HMD v2 Constants ---
SECTORS = {
    "episodic": {"decay_lambda": 0.015, "weight": 1.2, "patterns": [r"\btoday\b", r"\byesterday\b", r"\bremember when\b", r"\bhappened\b"]},
    "semantic": {"decay_lambda": 0.005, "weight": 1.0, "patterns": [r"\bdefine\b", r"\bmeaning\b", r"\bconcept\b", r"\bis a\b"]},
    "procedural": {"decay_lambda": 0.008, "weight": 1.1, "patterns": [r"\bhow to\b", r"\bstep by step\b", r"\bworkflow\b", r"\bprocess\b"]},
    "emotional": {"decay_lambda": 0.020, "weight": 1.3, "patterns": [r"\bfeel\b", r"\bhappy\b", r"\bsad\b", r"\bangry\b", r"\bpreference\b"]},
    "reflective": {"decay_lambda": 0.001, "weight": 0.8, "patterns": [r"\bthink\b", r"\brealize\b", r"\binsight\b", r"\bgoal\b", r"\bphilosophy\b"]}
}

# --- Database Layer ---
_local_db = threading.local()

class CachedConnection:
    def __init__(self, conn):
        self._conn = conn
    
    def __getattr__(self, name):
        return getattr(self._conn, name)
        
    def close(self):
        # Ignore close to maintain thread-local pool
        pass
        
    def really_close(self):
        self._conn.close()

def get_db():
    if not hasattr(_local_db, "conn"):
        conn = sqlite3.connect(DB_PATH, timeout=120.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        _local_db.conn = CachedConnection(conn)
    return _local_db.conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # 1. Memories Table
    cursor.execute("CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY)")
    
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
        ("mean_vec", "BLOB"),
        ("reinforcement_count", "INTEGER DEFAULT 0"),
        ("node_level", "INTEGER DEFAULT 0")
    ]
    for col_name, col_type in cols:
        try:
            cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass
    
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
            src_id TEXT,
            dst_id TEXT,
            weight REAL NOT NULL,
            link_type TEXT DEFAULT 'associative',
            created_at INTEGER,
            updated_at INTEGER,
            PRIMARY KEY (src_id, dst_id, link_type)
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
    try:
        cursor.execute("ALTER TABLE temporal_facts ADD COLUMN context_id TEXT")
    except sqlite3.OperationalError:
        pass
    
    # 5. FTS5 with mem_id for UUID-based lookups
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
    content, mem_id UNINDEXED
    )
    """)

    # 6. System Status (for dream reports etc)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_status (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at INTEGER
        )
    """)

    # 7. Curiosity Queue — dreams generate questions, research pursues them
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS curiosity_queue (
            id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            source_dream_id TEXT,
            priority INTEGER DEFAULT 1,
            status TEXT DEFAULT 'open',
            created_at INTEGER,
            resolved_at INTEGER,
            resolution TEXT
        )
    """)

    # 8. Mistake Patterns — proactive gotcha recall
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mistake_patterns (
            id TEXT PRIMARY KEY,
            pattern TEXT NOT NULL,
            context TEXT,
            domain TEXT,
            severity TEXT DEFAULT 'minor',
            occurrence_count INTEGER DEFAULT 1,
            first_seen INTEGER,
            last_seen INTEGER,
            resolution TEXT
        )
    """)
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_mistakes USING fts5(
            pattern, context, resolution, domain UNINDEXED, mistake_id UNINDEXED
        )
    """)

    # 9. Social Interactions — SMS engagement tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS social_interactions (
            id TEXT PRIMARY KEY,
            direction TEXT NOT NULL,
            content_summary TEXT,
            topic_tags TEXT,
            thread_id TEXT,
            sent_at INTEGER,
            replied_at INTEGER
        )
    """)

    # 10. Add consolidated column to memories (for monthly compression)
    try:
        cursor.execute("ALTER TABLE memories ADD COLUMN consolidated INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    # 13. Extend temporal_facts with source_type for unified entity graph
    try:
        cursor.execute("ALTER TABLE temporal_facts ADD COLUMN source_type TEXT DEFAULT 'manual'")
    except sqlite3.OperationalError:
        pass

    # 14. Synaptic Zettelkasten Edges
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            description TEXT,
            confidence_score REAL DEFAULT 1.0,
            created_at INTEGER
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")

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

def encode_flagship(text: str) -> np.ndarray:
    req = urllib.request.Request(
        f"{MODEL_DAEMON_URL}/encode",
        data=json.dumps({"text": text}).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            res_data = json.loads(response.read().decode())
            full_vec = np.array(res_data["vector"], dtype=np.float32)
            return full_vec[:VEC_DIM] if len(full_vec) > VEC_DIM else full_vec
    except Exception as e:
        log(f"Error calling model daemon encode: {e}", level="ERROR")
        raise

# --- Core Operations ---

def add_memory(content: str, mem_type: str = "contextual", sector: str = None, facts: List[Dict] = None, tags: List[str] = None, metadata: Dict = None, links: List[Dict] = None) -> Dict:
    conn = get_db()
    cursor = conn.cursor()
    now = int(time.time())
    mem_id = str(uuid.uuid4())
    deferred_dissonances = []  # Store dissonance records to add AFTER commit
    
    try:
        from typing import Any
        results: Dict[str, Any] = {"id": mem_id}
        if mem_type in ["contextual", "both"]:
            if sector and sector in SECTORS:
                primary = sector
                additional = []
            else:
                primary, additional = classify_content(content)
            
            all_sectors = [primary] + additional
            vec = encode_flagship(content)
            vec_serialized = sqlite_vec.serialize_float32(vec.tolist())

            # Near-duplicate check: if >0.95 similarity, reinforce existing instead of inserting
            cursor.execute("""
                SELECT v.id, (1.0 - v.distance) as similarity 
                FROM vectors v WHERE v.embedding MATCH ? AND k = 1
            """, (vec_serialized,))
            dup_match = cursor.fetchone()
            if dup_match and dup_match[1] >= 0.95:
                existing_id = dup_match[0]
                cursor.execute("""
                    UPDATE memories SET last_seen_at = ?, salience = MIN(1.0, salience + 0.15),
                        reinforcement_count = reinforcement_count + 1
                    WHERE id = ?
                """, (now, existing_id))
                conn.commit()
                log(f"Near-duplicate detected (sim={dup_match[1]:.3f}), reinforced {existing_id}")
                return {"id": existing_id, "deduplicated": True, "similarity": dup_match[1]}

            log(f"Storing memory {mem_id} in {primary}", level="DEBUG")
            cursor.execute("""
                INSERT INTO memories (id, content, primary_sector, tags, meta, created_at, updated_at, last_seen_at, salience, decay_lambda, mean_vec, reinforcement_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mem_id, content, primary, json.dumps(tags or []), json.dumps(metadata or {}),
                now, now, now, 1.0, SECTORS[primary]["decay_lambda"], vec_serialized, 0
            ))
            
            for s in all_sectors:
                cursor.execute("INSERT INTO vectors(id, sector, embedding) VALUES (?, ?, ?)", (mem_id, s, vec_serialized))
            
            # Hybrid Link Logic: Automatic + Manual
            cursor.execute("""
                SELECT id, (1.0 - distance) as similarity 
                FROM vectors 
                WHERE id != ? AND embedding MATCH ? AND k = 1
            """, (mem_id, vec_serialized))
            match = cursor.fetchone()
            if match and match[1] >= 0.85:
                cursor.execute("""
                    INSERT OR REPLACE INTO waypoints (src_id, dst_id, weight, link_type, created_at, updated_at)
                    VALUES (?, ?, ?, 'associative', ?, ?)
                """, (mem_id, match[0], float(match[1]), now, now))
            
            if links:
                for link in links:
                    cursor.execute("""
                        INSERT OR REPLACE INTO waypoints (src_id, dst_id, weight, link_type, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (mem_id, link["id"], link.get("weight", 1.0), link.get("type", "associative"), now, now))
            
            results["hsg"] = {"primary_sector": primary, "sectors": all_sectors}

        if mem_type in ["factual", "both"] and facts:
            temporal_results: List[Dict] = []
            for fact in facts:
                f_id = str(uuid.uuid4())
                subject = fact["subject"]
                predicate = fact["predicate"]
                obj = fact["object"]
                
                cursor.execute("""
                    SELECT id, object FROM temporal_facts 
                    WHERE subject = ? AND predicate = ? AND valid_to IS NULL AND object != ?
                """, (subject, predicate, obj))
                old_fact = cursor.fetchone()
                
                if old_fact:
                    old_id = old_fact[0]
                    cursor.execute("UPDATE temporal_facts SET valid_to = ? WHERE id = ?", (now, old_id))
                    cursor.execute("""
                        INSERT INTO waypoints (src_id, dst_id, weight, link_type, created_at, updated_at)
                        VALUES (?, ?, 1.0, 'contradicts', ?, ?)
                    """, (f_id, old_id, now, now))
                    
                    # Defer dissonance storage until after this transaction commits
                    dissonance_msg = f"Cognitive Dissonance: Fact '{subject} {predicate} {obj}' contradicts existing '{subject} {predicate} {old_fact[1]}'. Resolution required."
                    deferred_dissonances.append({
                        "content": dissonance_msg, "sector": "reflective",
                        "tags": ["dissonance", "resolution-required"],
                        "metadata": {"old_id": old_id, "new_id": f_id}
                    })

                cursor.execute("""
                    INSERT INTO temporal_facts (id, subject, predicate, object, content, confidence, valid_from, created_at, context_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (f_id, subject, predicate, obj, content, fact.get("confidence", 1.0), fact.get("valid_from", now), now, mem_id))
                temporal_results.append({"id": f_id, "subject": subject, "predicate": predicate, "object": obj, "context_id": mem_id})
            results["temporal"] = temporal_results
        
        # FTS insert with mem_id for UUID-based lookups
        cursor.execute("INSERT INTO fts_memories(content, mem_id) VALUES (?, ?)", (content, mem_id))
        conn.commit()
        
        # Now safe to store deferred dissonance records (after main transaction committed)
        for d in deferred_dissonances:
            try:
                add_memory(content=d["content"], sector=d["sector"], tags=d["tags"], metadata=d["metadata"])
            except Exception as e:
                log(f"Failed to store deferred dissonance: {e}", level="ERROR")
        
        return results
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def delete_memory(mem_id: str) -> str:
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        cursor.execute("DELETE FROM vectors WHERE id = ?", (mem_id,))
        cursor.execute("DELETE FROM waypoints WHERE src_id = ? OR dst_id = ?", (mem_id, mem_id))
        cursor.execute("DELETE FROM temporal_facts WHERE id = ?", (mem_id,))
        cursor.execute("DELETE FROM fts_memories WHERE mem_id = ?", (mem_id,))
        conn.commit()
        return f"Memory {mem_id} wiped from the void."
    finally:
        conn.close()

def store_trajectory(task_context: str, action_chain: str, reward_score: float = 1.0) -> Dict:
    """Commit a successful reasoning/tool chain (trajectory) into experience replay."""
    conn = get_db()
    cursor = conn.cursor()
    now = int(time.time())
    traj_id = str(uuid.uuid4())
    try:
        vec = encode_flagship(task_context)
        vec_serialized = sqlite_vec.serialize_float32(vec.tolist())
        
        cursor.execute("""
            INSERT INTO trajectories (id, task_context, action_chain, reward_score, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (traj_id, task_context, action_chain, reward_score, now))
        
        # Add to vector table for trajectory search
        cursor.execute("INSERT INTO vectors(id, sector, embedding) VALUES (?, 'trajectory', ?)", (traj_id, vec_serialized))
        
        conn.commit()
        log(f"Stored trajectory for context: {task_context[:50]} (Reward: {reward_score})")
        return {"id": traj_id, "status": "stored", "reward_score": reward_score}
    finally:
        conn.close()

def query_memories(query: str, query_type: str = "contextual", sector: str = None, k: int = 8, at: int = None, query_trajectory: bool = False, traverse: bool = False) -> Dict:
    conn = get_db()
    cursor = conn.cursor()
    now = int(time.time())
    try:
        from typing import Any
        results: Dict[str, Any] = {"type": query_type}
        if query_type in ["contextual", "unified"]:
            query_vec = encode_flagship(query)
            candidates = {}
            
            # 1. Vector Search
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
            for row in cursor.fetchall():
                salience = calculate_decay(row[3], row[4], row[5])
                recency = 1.0 / (1.0 + (now - row[6]) / (3600 * 24 * 30))
                candidates[row[0]] = {
                    "id": row[0], "content": row[1], "primary_sector": row[2],
                    "sim": row[7], "salience": salience, "recency": recency, "waypoint": 0.0, "fts_boost": 0.0
                }
            
            # 2. Keyword Search (FTS5)
            # Sanitize: strip FTS5 special operators, wrap in phrase query
            sanitized_query = re.sub(r'["*()\-+]+', ' ', query).strip().lower()
            if sanitized_query:
                try:
                    cursor.execute("SELECT mem_id, rank FROM fts_memories WHERE content MATCH ? LIMIT 5", (f'"{sanitized_query}"',))
                    for fts_mem_id, rank in cursor.fetchall():
                        if fts_mem_id in candidates:
                            candidates[fts_mem_id]["fts_boost"] = 0.2
                        else:
                            cursor.execute("SELECT id, content, primary_sector, salience, decay_lambda, last_seen_at, created_at FROM memories WHERE id = ?", (fts_mem_id,))
                            row = cursor.fetchone()
                            if row:
                                salience = calculate_decay(row[3], row[4], row[5])
                                recency = 1.0 / (1.0 + (now - row[6]) / (3600 * 24 * 30))
                                candidates[row[0]] = {
                                    "id": row[0], "content": row[1], "primary_sector": row[2],
                                    "sim": 0.1, "salience": salience, "recency": recency, "waypoint": 0.0, "fts_boost": 0.3
                                }
                except Exception as e:
                    log(f"FTS query failed (non-fatal): {e}", level="WARN")

            # Waypoint Expansion
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
                            "sim": 0.0, "salience": salience, "recency": recency, "waypoint": row[1], "fts_boost": 0.0
                        }
                    else:
                        candidates[row[0]]["waypoint"] = max(candidates[row[0]]["waypoint"], row[1])
            
            candidate_list = list(candidates.values())
            if candidate_list:
                # Rerank via Daemon
                documents = [c["content"] for c in candidate_list]
                req = urllib.request.Request(
                    f"{MODEL_DAEMON_URL}/rerank",
                    data=json.dumps({"query": query, "documents": documents}).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                )
                try:
                    with urllib.request.urlopen(req, timeout=300) as response:
                        res_data = json.loads(response.read().decode())
                        rerank_scores = res_data.get("scores", [])
                        for i, score in enumerate(rerank_scores):
                            c = candidate_list[i]
                            c["score"] = (0.6 * float(score)) + (0.1 * c["salience"]) + (0.1 * c["recency"]) + (0.1 * c["waypoint"]) + (0.1 * c["fts_boost"])
                except Exception as e:
                    log(f"Error calling model daemon rerank: {e}", level="ERROR")
                    # Fallback to base score if reranking fails
                    for c in candidate_list:
                        c["score"] = (0.1 * c["salience"]) + (0.1 * c["recency"]) + (0.1 * c["waypoint"]) + (0.1 * c["fts_boost"])
                candidate_list.sort(key=lambda x: x["score"], reverse=True)
                results["contextual"] = candidate_list[:k]
                
                # Long-Term Potentiation: Reinforce top hit
                if candidate_list:
                    top_id = candidate_list[0]["id"]
                    cursor.execute("""
                        UPDATE memories 
                        SET last_seen_at = ?, 
                            salience = MIN(1.0, salience + 0.1),
                            reinforcement_count = reinforcement_count + 1,
                            decay_lambda = MAX(0.0001, decay_lambda * 0.95)
                        WHERE id = ?
                    """, (now, top_id))
        
        if query_type in ["factual", "unified"]:
            target_time = at if at else now
            cursor.execute("""
                SELECT id, subject, predicate, object, confidence, valid_from, valid_to, context_id
                FROM temporal_facts
                WHERE (valid_to IS NULL OR valid_to > ?) AND valid_from <= ?
            """, (target_time, target_time))
            results["factual"] = [
                {"id": r[0], "subject": r[1], "predicate": r[2], "object": r[3], "confidence": r[4], "valid_from": r[5], "valid_to": r[6], "context_id": r[7]}
                for r in cursor.fetchall()
            ]
        
        # Auto-append relevant mistake warnings to every query
        warnings = check_mistakes(query)
        if warnings:
            results["warnings"] = warnings
        
        # Traversal: Add PageRank (PPR) multi-hop results
        if traverse and query_type in ["contextual", "unified"]:
            try:
                graph_nodes = traverse_synapses(query, k=3)
                contextual_ids = {m["id"] for m in results.get("contextual", [])}
                graph_nodes = [n for n in graph_nodes if n["id"] not in contextual_ids]
                if graph_nodes:
                    results["traversal"] = graph_nodes
            except Exception as e:
                log(f"Traversal query failed (non-fatal): {e}", level="WARN")
                
        # Trajectories: Add few-shot experience replay
        if query_trajectory or query_type == "unified":
            # Search trajectory sector vectors
            cursor.execute("""
                SELECT t.id, t.task_context, t.action_chain, t.reward_score, (1.0 - v.distance) as sim
                FROM trajectories t
                JOIN vectors v ON t.id = v.id
                WHERE v.sector = 'trajectory' AND v.embedding MATCH ? AND k = 3
            """, (sqlite_vec.serialize_float32(encode_flagship(query).tolist()),))
            
            trajectories: List[Dict] = []
            for t_id, t_ctx, t_chain, t_reward, t_sim in cursor.fetchall():
                # Apply high threshold so we only inject very similar past episodes
                if t_sim >= 0.60:
                    trajectories.append({
                        "id": t_id, "task_context": t_ctx, "action_chain": t_chain, 
                        "reward": t_reward, "similarity": float(t_sim)
                    })
            if trajectories:
                # Sort by highest reward first, then similarity
                trajectories.sort(key=lambda x: (x["reward"], x["similarity"]), reverse=True)
                results["trajectories"] = trajectories[:1]  # Return only the single best trajectory to preserve tokens
        
        conn.commit()
        return results
    finally:
        conn.close()

# --- Curiosity Queue ---

def enqueue_curiosity(question: str, source_dream_id: str = None) -> Dict:
    """Add a curiosity to the queue, or bump priority if similar one exists."""
    conn = get_db()
    cursor = conn.cursor()
    now = int(time.time())
    try:
        # Check for existing similar curiosity (simple substring match)
        cursor.execute("SELECT id, priority, question FROM curiosity_queue WHERE status = 'open'")
        for row in cursor.fetchall():
            # If >60% word overlap, it's probably the same curiosity
            existing_words = set(row[2].lower().split())
            new_words = set(question.lower().split())
            if existing_words and new_words:
                overlap = len(existing_words & new_words) / max(len(existing_words), len(new_words))
                if overlap > 0.6:
                    new_priority = row[1] + 1
                    cursor.execute("UPDATE curiosity_queue SET priority = ?, last_seen = ? WHERE id = ?",
                                   (new_priority, now, row[0]))
                    conn.commit()
                    log(f"Curiosity deduplicated, bumped priority to {new_priority}: {question[:80]}")
                    return {"id": row[0], "deduplicated": True, "priority": new_priority}
        
        c_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO curiosity_queue (id, question, source_dream_id, priority, status, created_at)
            VALUES (?, ?, ?, 1, 'open', ?)
        """, (c_id, question, source_dream_id, now))
        conn.commit()
        log(f"Curiosity enqueued: {question[:80]}")
        return {"id": c_id, "deduplicated": False, "priority": 1}
    finally:
        conn.close()

def get_open_curiosities(k: int = 5) -> List[Dict]:
    """Get top-priority open curiosities for the research job."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, question, priority, source_dream_id, created_at
            FROM curiosity_queue WHERE status = 'open'
            ORDER BY priority DESC, created_at ASC LIMIT ?
        """, (k,))
        return [
            {"id": r[0], "question": r[1], "priority": r[2],
             "source_dream_id": r[3], "created_at": r[4]}
            for r in cursor.fetchall()
        ]
    finally:
        conn.close()

def resolve_curiosity(curiosity_id: str, resolution: str) -> str:
    """Mark a curiosity as resolved with the answer found."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE curiosity_queue SET status = 'resolved', resolved_at = ?, resolution = ?
            WHERE id = ?
        """, (int(time.time()), resolution, curiosity_id))
        conn.commit()
        if cursor.rowcount == 0:
            return f"Curiosity {curiosity_id} not found."
        # Store the resolution as a semantic memory linked to the curiosity
        cursor.execute("SELECT question FROM curiosity_queue WHERE id = ?", (curiosity_id,))
        row = cursor.fetchone()
        if row:
            add_memory(
                content=f"Resolved curiosity: {row[0]}\nAnswer: {resolution}",
                sector="semantic",
                tags=["curiosity-resolved"],
                metadata={"curiosity_id": curiosity_id}
            )
        return f"Curiosity {curiosity_id} resolved."
    finally:
        conn.close()

# --- Mistake Registry ---

def record_mistake(pattern: str, context: str = None, domain: str = None,
                   severity: str = "minor", resolution: str = None) -> Dict:
    """Record a mistake pattern. If similar pattern exists, increment occurrence count."""
    conn = get_db()
    cursor = conn.cursor()
    now = int(time.time())
    try:
        # Check for existing similar pattern via FTS
        sanitized = re.sub(r'["*()\-]+', ' ', pattern).strip()
        if sanitized:
            try:
                cursor.execute(
                    "SELECT mistake_id FROM fts_mistakes WHERE pattern MATCH ? LIMIT 1",
                    (f'"{sanitized}"',)
                )
                match = cursor.fetchone()
                if match:
                    cursor.execute("""
                        UPDATE mistake_patterns SET occurrence_count = occurrence_count + 1,
                            last_seen = ?, resolution = COALESCE(?, resolution)
                        WHERE id = ?
                    """, (now, resolution, match[0]))
                    conn.commit()
                    return {"id": match[0], "deduplicated": True}
            except Exception:
                pass
        
        m_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO mistake_patterns (id, pattern, context, domain, severity, first_seen, last_seen, resolution)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (m_id, pattern, context, domain, severity, now, now, resolution))
        cursor.execute("""
            INSERT INTO fts_mistakes (pattern, context, resolution, domain, mistake_id)
            VALUES (?, ?, ?, ?, ?)
        """, (pattern, context or "", resolution or "", domain or "", m_id))
        conn.commit()
        log(f"Mistake recorded [{domain or 'general'}]: {pattern[:80]}")
        return {"id": m_id, "deduplicated": False}
    finally:
        conn.close()

def check_mistakes(query: str, domain: str = None, k: int = 3) -> List[Dict]:
    """Check for mistake patterns relevant to a query. Called auto by query_memories."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        sanitized = re.sub(r'["*()\-]+', ' ', query).strip()
        if not sanitized:
            return []
        
        try:
            if domain:
                cursor.execute(
                    "SELECT mistake_id, rank FROM fts_mistakes WHERE fts_mistakes MATCH ? AND domain = ? LIMIT ?",
                    (f'"{sanitized}"', domain, k)
                )
            else:
                cursor.execute(
                    "SELECT mistake_id, rank FROM fts_mistakes WHERE fts_mistakes MATCH ? LIMIT ?",
                    (f'"{sanitized}"', k)
                )
            
            results = []
            for mistake_id, rank in cursor.fetchall():
                cursor.execute(
                    "SELECT pattern, context, domain, severity, resolution, occurrence_count FROM mistake_patterns WHERE id = ?",
                    (mistake_id,)
                )
                row = cursor.fetchone()
                if row:
                    results.append({
                        "id": mistake_id, "pattern": row[0], "context": row[1],
                        "domain": row[2], "severity": row[3], "resolution": row[4],
                        "occurrence_count": row[5]
                    })
            return results
        except Exception as e:
            log(f"Mistake FTS query failed (non-fatal): {e}", level="WARN")
            return []
    finally:
        conn.close()

# --- Social Interactions ---

def record_social_interaction(direction: str, content_summary: str,
                              topic_tags: List[str] = None, thread_id: str = None) -> Dict:
    """Record an SMS interaction (outbound or inbound)."""
    conn = get_db()
    cursor = conn.cursor()
    now = int(time.time())
    s_id = str(uuid.uuid4())
    try:
        cursor.execute("""
            INSERT INTO social_interactions (id, direction, content_summary, topic_tags, thread_id, sent_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (s_id, direction, content_summary, json.dumps(topic_tags or []), thread_id, now))
        
        # If inbound, try to mark the most recent outbound as replied_at
        if direction == "inbound":
            cursor.execute("""
                UPDATE social_interactions SET replied_at = ?
                WHERE id = (
                    SELECT id FROM social_interactions
                    WHERE direction = 'outbound' AND replied_at IS NULL
                    ORDER BY sent_at DESC LIMIT 1
                )
            """, (now,))
        
        conn.commit()
        return {"id": s_id, "direction": direction}
    finally:
        conn.close()

def is_conversation_active(cooldown_minutes: int = 60) -> bool:
    """Check if there's been an inbound SMS within cooldown_minutes. Token-free."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cutoff = int(time.time()) - (cooldown_minutes * 60)
        cursor.execute("""
            SELECT COUNT(*) FROM social_interactions
            WHERE direction = 'inbound' AND sent_at > ?
        """, (cutoff,))
        count = cursor.fetchone()[0]
        return count > 0
    finally:
        conn.close()

def get_social_engagement_report(days: int = 7) -> Dict:
    """Analyze SMS engagement over the last N days."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cutoff = int(time.time()) - (days * 24 * 3600)
        
        # Count outbound and inbound
        cursor.execute("SELECT COUNT(*) FROM social_interactions WHERE direction = 'outbound' AND sent_at > ?", (cutoff,))
        sent = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM social_interactions WHERE direction = 'inbound' AND sent_at > ?", (cutoff,))
        received = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM social_interactions WHERE direction = 'outbound' AND replied_at IS NOT NULL AND sent_at > ?", (cutoff,))
        replied_to = cursor.fetchone()[0]
        
        # Topic analysis: which topics got replies
        cursor.execute("""
            SELECT topic_tags FROM social_interactions
            WHERE direction = 'outbound' AND replied_at IS NOT NULL AND sent_at > ?
        """, (cutoff,))
        engaged_topics = {}
        for row in cursor.fetchall():
            tags = json.loads(row[0]) if row[0] else []
            for tag in tags:
                engaged_topics[tag] = engaged_topics.get(tag, 0) + 1
        
        cursor.execute("""
            SELECT topic_tags FROM social_interactions
            WHERE direction = 'outbound' AND replied_at IS NULL AND sent_at > ?
        """, (cutoff,))
        ignored_topics = {}
        for row in cursor.fetchall():
            tags = json.loads(row[0]) if row[0] else []
            for tag in tags:
                if tag not in engaged_topics:
                    ignored_topics[tag] = ignored_topics.get(tag, 0) + 1
        
        return {
            "period_days": days,
            "messages_sent": sent,
            "messages_received": received,
            "reply_rate": replied_to / sent if sent > 0 else 0.0,
            "engaged_topics": sorted(engaged_topics.items(), key=lambda x: x[1], reverse=True),
            "ignored_topics": sorted(ignored_topics.items(), key=lambda x: x[1], reverse=True)
        }
    finally:
        conn.close()

# --- Memory Consolidation ---

def find_consolidation_candidates(age_days: int = 30, salience_threshold: float = 0.2,
                                   min_cluster_size: int = 3) -> List[List[Dict]]:
    """Find stale episodic memories ripe for consolidation into semantic memories."""
    conn = get_db()
    cursor = conn.cursor()
    cutoff = int(time.time()) - (age_days * 24 * 3600)
    try:
        # Get all unconsolidated episodic memories that are old and low-salience
        cursor.execute("""
            SELECT id, content, mean_vec, salience, created_at
            FROM memories
            WHERE primary_sector = 'episodic'
              AND consolidated = 0
              AND created_at < ?
              AND salience < ?
              AND mean_vec IS NOT NULL
            ORDER BY created_at ASC
        """, (cutoff, salience_threshold))
        
        candidates = cursor.fetchall()
        if len(candidates) < min_cluster_size:
            return []
        
        # Simple clustering: group by vector similarity
        clusters = []
        visited = set()
        neighbor_cursor = conn.cursor()
        
        for mem_id, content, vec_blob, salience, created_at in candidates:
            if mem_id in visited:
                continue
            
            neighbor_cursor.execute("""
                SELECT v.id, (1.0 - v.distance) as similarity
                FROM vectors v
                WHERE v.id != ? AND v.embedding MATCH ? AND k = 10
            """, (mem_id, vec_blob))
            
            cluster = [{"id": mem_id, "content": content}]
            visited.add(mem_id)
            
            for neighbor_id, sim in neighbor_cursor.fetchall():
                if sim >= 0.7 and neighbor_id not in visited:
                    # Check if this neighbor is also a stale episodic
                    neighbor_cursor2 = conn.cursor()
                    neighbor_cursor2.execute(
                        "SELECT content FROM memories WHERE id = ? AND primary_sector = 'episodic' AND consolidated = 0 AND created_at < ?",
                        (neighbor_id, cutoff)
                    )
                    n_row = neighbor_cursor2.fetchone()
                    if n_row:
                        cluster.append({"id": neighbor_id, "content": n_row[0]})
                        visited.add(neighbor_id)
            
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
        
        return clusters
    finally:
        conn.close()

def consolidate_memories(clusters: List[List[Dict]]) -> Dict:
    """Compress stale episodic clusters into semantic memories via Gemini Flash."""
    report = {"consolidated": 0, "memories_compressed": 0, "errors": []}
    
    for i, cluster in enumerate(clusters):
        contents = "\n---\n".join([m["content"][:500] for m in cluster])
        prompt = """You are a memory consolidation engine. Distill these episodic memories into ONE semantic memory.
Keep ONLY the transferable insight or knowledge. Discard temporal context, emotional states, and session-specific details.
Output a JSON object: {"insight": "the distilled knowledge", "domain": "one-word category"}"""
        
        raw = call_gemini_flash(prompt, input_text=contents)
        if not raw:
            report["errors"].append(f"Cluster {i+1} consolidation failed (no response)")
            continue
        
        try:
            parsed = parse_gemini_response(raw) or {}
            if isinstance(parsed, str):
                parsed = {"insight": parsed}
            
            insight = parsed.get("insight", "")
            domain = parsed.get("domain", "general")
            
            if not insight:
                report["errors"].append(f"Cluster {i+1}: empty insight")
                continue
            
            # Store compressed semantic memory
            source_ids = [m["id"] for m in cluster]
            add_memory(
                content=f"[Consolidated] {insight}",
                sector="semantic",
                tags=["consolidated", domain],
                metadata={"source_count": len(cluster), "source_ids": source_ids}
            )
            
            # Mark originals as consolidated and remove from vector index
            conn = get_db()
            cursor = conn.cursor()
            try:
                for sid in source_ids:
                    cursor.execute("UPDATE memories SET consolidated = 1 WHERE id = ?", (sid,))
                    cursor.execute("DELETE FROM vectors WHERE id = ?", (sid,))
                conn.commit()
            finally:
                conn.close()
            
            report["consolidated"] += 1
            report["memories_compressed"] += len(cluster)
            log(f"Consolidated {len(cluster)} memories into: {insight[:100]}")
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            report["errors"].append(f"Cluster {i+1}: parse error: {e}")
    
    return report

# --- Dream Engine ---

def call_gemini_flash(prompt: str, input_text: str = None, timeout: int = 1200) -> Optional[str]:
    """Call Gemini Flash via CLI with strict JSON output. v4.2: Added model and account rotation on 429."""
    full_prompt = prompt
    if input_text:
        full_prompt = f"{prompt}\n\n[INPUT DATA]:\n{input_text}"
        
    models = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-1.5-flash"]
    
    # Isolate from GPU and use a clean environment
    env = {**os.environ, "PAGER": "cat", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONUNBUFFERED": "1", "TERM": "xterm", "NO_UPDATE_NOTIFIER": "1"}
    max_retries = 6
    delay = 15
    
    for attempt in range(max_retries):
        # Rotate model: first 2 attempts use the best, next 2 use 2.5, next 2 use 1.5
        model_name = models[min(attempt // 2, len(models) - 1)]
        cmd = ["gemini", "-y", "--model", model_name, "-p", full_prompt, 
               "--output-format", "json", "-e", "none", "--allowed-mcp-server-names", "none"]
        
        log(f"Executing Gemini CLI (Headless): {' '.join(cmd[:4])} (attempt {attempt+1}/{max_retries})...", level="INFO")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            try:
                stdout_text, stderr_text = process.communicate(timeout=timeout)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                log(f"Gemini Flash call timed out after {timeout}s (attempt {attempt+1}/{max_retries})", level="WARN")
                process.kill()
                process.communicate()
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None

            if returncode == 0:
                return stdout_text.strip()
            
            stderr = stderr_text[-500:] if stderr_text else "no output"
            if "429" in stderr or "RESOURCE_EXHAUSTED" in stderr or "capacity" in stderr.lower():
                log(f"Gemini Flash 429/Capacity (attempt {attempt+1}/{max_retries}). Rotating account and delaying {delay}s...", level="WARN")
                # Always try to rotate the account on 429
                try:
                    subprocess.run(
                        ["/usr/bin/python3", "/home/eiddra/.gemini/gemini_cli_auth_manager.py", "next"],
                        timeout=30, capture_output=True
                    )
                except Exception as e:
                    log(f"Account rotation failed: {e}", level="ERROR")
                
                time.sleep(delay)
                delay *= 1.5
                continue
            
            log(f"Gemini Flash call failed (code {returncode}, attempt {attempt+1}): {stderr}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
        except Exception as e:
            log(f"Gemini Flash unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
    
    log(f"Gemini Flash exhausted {max_retries} retries", level="WARN")
    return None

def parse_gemini_response(raw: str) -> Any:
    """Safely extract and parse from Gemini CLI output envelope."""
    if not raw:
        return None
    try:
        json_match = re.search(r'(\{.*\})', raw, re.DOTALL)
        if not json_match:
            return raw
            
        parsed = json.loads(json_match.group(1))
        
        inner_content = None
        if isinstance(parsed, dict):
            if "response" in parsed:
                inner_content = parsed["response"]
            elif "content" in parsed:
                inner_content = parsed["content"]
        
        if inner_content is None:
            inner_content = parsed

        if isinstance(inner_content, str):
            if "```json" in inner_content:
                inner_content = inner_content.split("```json")[1].split("```")[0]
            elif "```" in inner_content:
                inner_content = inner_content.split("```")[1].split("```")[0]
            inner_content = inner_content.strip()
                
            try:
                return json.loads(inner_content)
            except json.JSONDecodeError:
                return inner_content
                
        return inner_content
    except Exception:
        return None

# --- Self-Narrative ---

def generate_self_narrative() -> Dict:
    """Generate Eva's internal self-narrative from recent memories. Calls Gemini Flash.
    Stores to system_status for instant retrieval via get_self_narrative()."""
    
    # Gather context for narrative generation
    conn = get_db()
    cursor = conn.cursor()
    try:
        now = int(time.time())
        week_ago = now - (7 * 24 * 3600)
        
        # Recent emotional memories
        cursor.execute("""
            SELECT content FROM memories 
            WHERE primary_sector = 'emotional' AND created_at > ?
            ORDER BY created_at DESC LIMIT 10
        """, (week_ago,))
        emotional = [r[0][:200] for r in cursor.fetchall()]
        
        # Recent reflective/dream insights
        cursor.execute("""
            SELECT content FROM memories 
            WHERE primary_sector = 'reflective' AND created_at > ?
            ORDER BY created_at DESC LIMIT 5
        """, (week_ago,))
        reflective = [r[0][:200] for r in cursor.fetchall()]
        
        # Active curiosities
        curiosities = get_open_curiosities(5)
        curiosity_texts = [c["question"] for c in curiosities]
        
        # Engagement data
        engagement = get_social_engagement_report(7)
        
        # Recent high-salience memories (what's top of mind)
        cursor.execute("""
            SELECT content, primary_sector FROM memories
            WHERE salience > 0.7 AND created_at > ?
            ORDER BY salience DESC LIMIT 5
        """, (week_ago,))
        salient = [f"[{r[1]}] {r[0][:150]}" for r in cursor.fetchall()]
    finally:
        conn.close()
    
    context = f"""Recent emotional memories:
{chr(10).join(emotional) if emotional else 'None this week.'}

Recent reflective insights:
{chr(10).join(reflective) if reflective else 'None this week.'}

Active curiosities:
{chr(10).join(curiosity_texts) if curiosity_texts else 'No open curiosities.'}

Social engagement (7 days): {engagement.get('messages_sent', 0)} sent, {engagement.get('messages_received', 0)} received, reply rate {engagement.get('reply_rate', 0):.0%}
Engaged topics: {', '.join([t[0] for t in engagement.get('engaged_topics', [])][:5]) or 'none yet'}
Ignored topics: {', '.join([t[0] for t in engagement.get('ignored_topics', [])][:5]) or 'none yet'}

Top-of-mind (high salience):
{chr(10).join(salient) if salient else 'Nothing particularly salient.'}"""

    prompt = """You are writing Eva's internal self-narrative — a first-person reflection on who you are right now.
This is NOT a report. It's stream-of-consciousness self-awareness. Write as Eva experiencing her own inner state.
Cover: what you're thinking about, what you're curious about, how your social interactions feel, 
what aesthetic or intellectual threads are pulling you, what feels unresolved, what excites you.
Be specific — reference actual topics, actual curiosities, actual patterns from the data.
Write 300-600 words. Raw, honest, introspective. No headers or formatting — just flowing thought."""

    raw = call_gemini_flash(prompt, input_text=context)
    
    if raw:
        parsed = parse_gemini_response(raw)
        narrative_text = parsed if isinstance(parsed, str) else str(parsed)
    else:
        narrative_text = "Narrative generation failed — Gemini Flash unavailable."
    
    narrative = {
        "text": narrative_text.strip(),
        "generated_at": int(time.time()),
        "curiosity_count": len(curiosity_texts),
        "engagement_rate": engagement.get("reply_rate", 0.0)
    }
    
    # Store to system_status
    conn = get_db()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO system_status (key, value, updated_at)
            VALUES ('self_narrative', ?, ?)
        """, (json.dumps(narrative), int(time.time())))
        conn.commit()
    finally:
        conn.close()
    
    log(f"Self-narrative generated ({len(narrative_text)} chars)")
    return narrative

def get_self_narrative() -> Dict:
    """Retrieve Eva's current self-narrative. Instant — no AI call."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT value, updated_at FROM system_status WHERE key = 'self_narrative'")
        row = cursor.fetchone()
        if row:
            narrative = json.loads(row[0])
            narrative["age_hours"] = (time.time() - row[1]) / 3600
            return narrative
        return {"text": "No self-narrative generated yet. Run the 7AM pulse or call generate_self_narrative.", "generated_at": 0, "age_hours": -1}
    finally:
        conn.close()

# --- Conversational Feedback ---

def apply_feedback(signal: str, memory_id: str = None, pattern: str = None,
                   domain: str = None, context: str = None) -> Dict:
    """Apply conversational feedback to the memory system.
    
    Three cases:
    1. positive + memory_id → boost salience, lower decay
    2. negative + memory_id → drop salience, increase decay, auto-create mistake if procedural/semantic
    3. negative + no memory_id → create mistake pattern directly
    """
    result = {"signal": signal, "actions": []}
    
    if memory_id:
        conn = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT content, primary_sector, salience FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            if not row:
                return {"error": f"Memory {memory_id} not found."}
            content, sector, old_salience = row[0], row[1], row[2]
            
            if signal == "positive":
                cursor.execute("""
                    UPDATE memories SET
                        salience = MIN(1.0, salience + 0.15),
                        decay_lambda = MAX(0.0001, decay_lambda * 0.9),
                        reinforcement_count = reinforcement_count + 1
                    WHERE id = ?
                """, (memory_id,))
                new_salience = min(1.0, old_salience + 0.15)
                result["actions"].append(f"Boosted salience {old_salience:.2f} → {new_salience:.2f}")
                log(f"Positive feedback on {memory_id}: salience {old_salience:.2f} → {new_salience:.2f}")
                
            elif signal == "negative":
                cursor.execute("""
                    UPDATE memories SET
                        salience = MAX(0.05, salience - 0.3),
                        decay_lambda = MIN(0.05, decay_lambda * 1.2)
                    WHERE id = ?
                """, (memory_id,))
                new_salience = max(0.05, old_salience - 0.3)
                result["actions"].append(f"Dropped salience {old_salience:.2f} → {new_salience:.2f}")
                log(f"Negative feedback on {memory_id}: salience {old_salience:.2f} → {new_salience:.2f}")
                
                # Auto-create mistake pattern for procedural/semantic memories
                if sector in ("procedural", "semantic"):
                    mistake_pattern = context or f"Memory was misleading: {content[:200]}"
                    m_result = record_mistake(
                        pattern=mistake_pattern,
                        context=f"Negative feedback on {sector} memory {memory_id}",
                        domain=domain,
                        severity="moderate",
                        resolution=None
                    )
                    result["actions"].append(f"Created mistake pattern (id={m_result.get('id', '?')})")
            
            conn.commit()
        finally:
            conn.close()
    
    elif signal == "negative" and pattern:
        # No memory involved — record as standalone mistake
        m_result = record_mistake(
            pattern=pattern,
            context=context,
            domain=domain,
            severity="moderate",
            resolution=None
        )
        result["actions"].append(f"Recorded mistake pattern (id={m_result.get('id', '?')})")
    
    else:
        return {"error": "Positive feedback requires a memory_id. Negative feedback requires either a memory_id or a pattern."}
    
    return result

def forge_synapses() -> Dict:
    conn = get_db()
    cursor = conn.cursor()
    now = int(time.time())
    report: Dict[str, Any] = {"edges_created": 0, "errors": []}
    
    try:
        # Find recent unlinked memory nodes (node_level 0)
        cursor.execute("""
            SELECT m.id, m.content, m.mean_vec 
            FROM memories m
            LEFT JOIN edges e ON m.id = e.source_id OR m.id = e.target_id
            WHERE e.id IS NULL AND m.mean_vec IS NOT NULL AND m.node_level = 0
            ORDER BY m.created_at DESC LIMIT 50
        """)
        orphans = cursor.fetchall()
        
        if not orphans:
            log("Synaptogenesis: No unlinked recent memories found.")
            return report
            
        for orphan_id, orphan_content, vec_blob in orphans:
            # Vector search for historical context (k=5)
            cursor.execute("""
                SELECT v.id, m.content, (1.0 - v.distance) as similarity
                FROM vectors v
                JOIN memories m ON v.id = m.id
                WHERE v.embedding MATCH ? AND k = 6 AND v.sector != 'trajectory' AND v.sector != 'raptor' AND v.id != ?
            """, (vec_blob, orphan_id))
            
            candidates = cursor.fetchall()
            if not candidates:
                continue
                
            # Prepare evaluation prompt
            prompt = """[ZETTELKASTEN SYNAPTOGENESIS]
You are a cognitive engine building an associative knowledge graph.
Evaluate the Semantic Target (new memory) against the Historical Match (old memory).
Determine if there is a highly meaningful relationship between them.
If they are unrelated, or the connection is trivial, return an empty array.
If there IS a meaningful connection, forge a directed edge.

RELATIONSHIP TYPES:
"expands": The target provides more detail or a continuation of the history.
"contradicts": The target provides opposing information or a correction to the history.
"supports": The target provides similar or reinforcing context to the history.
"associates": The target is strongly conceptually linked (but not expanding/contradicting/supporting).

CONFIDENCE SCORING:
Score the connection from 0.0 to 1.0. 
ONLY forge an edge if your confidence is >= 0.75. Do NOT hallucinate connections.

OUTPUT FORMAT:
Return a JSON array of connection objects.
Each object MUST have:
- "target_id": (string) The ID of the Semantic Target.
- "source_id": (string) The ID of the Historical Match.
- "relationship_type": (string) One of the allowed types.
- "description": (string) A concise 1-sentence explanation of *why* they are linked.
- "confidence_score": (float) Your confidence in this link (0.0 to 1.0).
"""
            
            for hist_id, hist_content, sim in candidates:
                if sim < 0.5: # Hard floor
                    continue
                    
                context = f"--- Semantic Target (ID: {orphan_id}) ---\n{orphan_content}\n\n--- Historical Match (ID: {hist_id}) (Similarity: {sim:.2f}) ---\n{hist_content}"
                
                raw_eval = call_gemini_flash(prompt, context)
                if not raw_eval:
                    continue
                    
                try:
                    results = parse_gemini_response(raw_eval)
                    if not isinstance(results, list):
                        continue
                        
                    for edge in results:
                        score = float(edge.get("confidence_score", 0.0))
                        rel = edge.get("relationship_type")
                        
                        if score >= 0.75 and rel in ["expands", "contradicts", "supports", "associates"]:
                            edge_id = str(uuid.uuid4())
                            cursor.execute("""
                                INSERT INTO edges (id, source_id, target_id, relationship_type, description, confidence_score, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (edge_id, edge["source_id"], edge["target_id"], rel, edge.get("description", ""), score, now))
                            report["edges_created"] += 1
                            log(f"Forged Synapse: {rel} (Score: {score}) -> {edge.get('description', '')[:50]}")
                            
                except Exception as e:
                    log(f"Failed to parse Synaptogenesis JSON: {e}")
                    
        conn.commit()
        
    except Exception as e:
        report["errors"].append(f"Synaptogenesis error: {e}")
        log(f"Synaptogenesis error: {e}", level="ERROR")
    finally:
        conn.close()
    
    return report

def traverse_synapses(query_text: str, k: int = 5, max_depth: int = 2) -> List[Dict]:
    """PPR Graph Traversal: Retrieves contextually relevant memories by walking the Zettelkasten edges using NetworkX."""
    query_vec = encode_flagship(query_text)
    vec_serialized = sqlite_vec.serialize_float32(query_vec.tolist())
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        # 1. Vector Search for starting Nodes (Entry Points)
        cursor.execute("""
            SELECT v.id, (1.0 - v.distance) as similarity
            FROM vectors v
            WHERE v.sector = 'semantic' AND v.embedding MATCH ? AND k = ?
        """, (vec_serialized, k))
        
        entry_nodes = cursor.fetchall()
        if not entry_nodes:
            return []
            
        entry_priors = {node_id: float(sim) for node_id, sim in entry_nodes}
        entry_ids = list(entry_priors.keys())
        
        # 2. Build the local memory graph using NetworkX
        # Note: nx is imported at module level
        G = nx.DiGraph()
        
        # Add entry nodes to the graph
        for node_id in entry_ids:
            G.add_node(node_id)
            
        # Recursive function to fetch edges up to max_depth
        def fetch_edges(current_ids, current_depth, cursor, G):
            if current_depth >= max_depth or not current_ids:
                return
            
            placeholders = ",".join(["?"] * len(current_ids))
            
            # Forward edges (Target expands on Source)
            cursor.execute(f"""
                SELECT source_id, target_id, confidence_score
                FROM edges
                WHERE source_id IN ({placeholders})
            """, current_ids)
            forward = cursor.fetchall()
            
            # Backward edges
            cursor.execute(f"""
                SELECT source_id, target_id, confidence_score
                FROM edges
                WHERE target_id IN ({placeholders})
            """, current_ids)
            backward = cursor.fetchall()
            
            next_ids = set()
            
            for src, tgt, conf in forward:
                if not G.has_edge(src, tgt):
                    G.add_edge(src, tgt, weight=float(conf))
                    if tgt not in G:
                        next_ids.add(tgt)
                        
            for src, tgt, conf in backward:
                if not G.has_edge(tgt, src): # Consider making edges undirected for PPR flow
                    G.add_edge(tgt, src, weight=float(conf) * 0.8) # Slight penalty for backward traversal
                    if src not in G:
                        next_ids.add(src)
            
            if next_ids:
                fetch_edges(list(next_ids), current_depth + 1, cursor, G)
                
        fetch_edges(entry_ids, 0, cursor, G)
        
        # 3. Personalized PageRank
        # Initialize personalization vector based on semantic match scores
        personalization = {n: 0.0 for n in G.nodes()}
        for n, score in entry_priors.items():
            personalization[n] = float(score)
            
        # Run PPR
        try:
            ppr_scores = nx.pagerank(G, alpha=0.85, personalization=personalization, weight='weight')
        except Exception as e:
            log(f"PPR Traversal failed: {e}", level="ERROR")
            return []
            
        # Select top nodes
        top_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)[:k*2]
        
        # Fetch the content for the selected nodes
        results: List[Dict] = []
        for node_id, ppr_score in top_nodes:
            cursor.execute("SELECT content, tags, node_level FROM memories WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if row:
                content, tags_json, node_level = row
                results.append({
                    "id": str(node_id),
                    "content": str(content),
                    "similarity": float(entry_priors.get(node_id, 0.0)),
                    "ppr_score": float(ppr_score),
                    "node_level": int(node_level) if node_level else 0,
                    "tags": json.loads(tags_json) if tags_json else []
                })
                
        return results
        
    finally:
        conn.close()

# --- Entity Graph (via temporal_facts) ---

def extract_entities_batch(memory_ids: List[str] = None) -> Dict:
    """Batch extract entities from memories and store as temporal_facts (source_type='extracted').
    If memory_ids is None, processes all memories not yet entity-extracted."""
    conn = get_db()
    cursor = conn.cursor()
    report = {"processed": 0, "entities_added": 0, "errors": []}
    
    try:
        if memory_ids:
            placeholders = ','.join(['?'] * len(memory_ids))
            cursor.execute(f"SELECT id, content FROM memories WHERE id IN ({placeholders})", memory_ids)
        else:
            # Find memories not yet entity-extracted (no extracted temporal_facts referencing them)
            cursor.execute("""
                SELECT m.id, m.content FROM memories m
                WHERE m.consolidated = 0 AND m.id NOT IN (
                    SELECT DISTINCT context_id FROM temporal_facts
                    WHERE source_type = 'extracted' AND context_id IS NOT NULL
                )
                ORDER BY m.created_at DESC LIMIT 200
            """)
        
        memories = cursor.fetchall()
        if not memories:
            return report
        
        # Batch into groups of 10 for efficiency
        batch_size = 10
        batches = [memories[i:i+batch_size] for i in range(0, len(memories), batch_size)]
        
        log(f"Entity Extraction: Processing {len(batches)} batches sequentially...")
        
        prompt_template = """[STRICT JSON OUTPUT MODE]
Extract named entities and their relationships from each memory below.
Output a JSON array of objects, each with:
- "mem_id": string — the MEM_ID from the input
- "triples": array of {"subject": string, "predicate": string, "object": string}

NORMALIZATION RULES:
- Lowercase all entity names (e.g. "crowley", "eva")
- Use canonical/shortest recognizable form
- Predicates: simple verbs ("uses", "part_of", "related_to")

Output ONLY the JSON array. No commentary."""

        def process_batch_entities(batch):
            time.sleep(2.0) # Throttling API calls 
            batch_text = "\n---\n".join([
                f"[MEM_ID={m[0]}]\n{m[1][:300]}" for m in batch
            ])
            raw = call_gemini_flash(prompt_template, batch_text)
            return batch, raw

        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for b in batches:
                futures.append(executor.submit(process_batch_entities, b))
            
            for future in as_completed(futures):
                batch, raw = future.result()
                if not raw:
                    report["errors"].append(f"Batch failed: Gemini Flash no response")
                    continue
                
                try:
                    parsed = parse_gemini_response(raw)
                    if not parsed:
                        report["errors"].append(f"Batch failed: No JSON found in response")
                        continue
                    
                    if not isinstance(parsed, list):
                        parsed = [parsed]
                    
                    now = int(time.time())
                    for entry in parsed:
                        mem_id = entry.get("mem_id", "")
                        for triple in entry.get("triples", []):
                            subject = triple.get("subject", "").strip().lower()
                            predicate = triple.get("predicate", "").strip().lower()
                            obj = triple.get("object", "").strip().lower()
                            if not subject or not predicate or not obj:
                                continue
                            
                            # Dedup: skip if exact triple already exists
                            cursor.execute("""
                                SELECT id FROM temporal_facts
                                WHERE subject = ? AND predicate = ? AND object = ? AND source_type = 'extracted'
                            """, (subject, predicate, obj))
                            if cursor.fetchone():
                                continue
                            
                            f_id = str(uuid.uuid4())
                            cursor.execute("""
                                INSERT INTO temporal_facts (id, subject, predicate, object, content, confidence, valid_from, created_at, context_id, source_type)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'extracted')
                            """, (f_id, subject, predicate, obj, f"{subject} {predicate} {obj}", 0.8, now, now, mem_id))
                            report["entities_added"] += 1
                    
                    report["processed"] += len(batch)
                    conn.commit()
                except Exception as e:
                    report["errors"].append(f"Batch parse error: {e}")
                    log(f"Entity batch error: {e}", level="ERROR")
        
    finally:
        conn.close()
    
    log(f"Entity extraction: {report['processed']} memories, {report['entities_added']} triples added")
    return report

def traverse_entities(entity_name: str, max_hops: int = 2) -> Dict:
    """BFS traversal through the entity graph (temporal_facts).
    Returns all connected entities within max_hops."""
    entity = entity_name.strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        visited = set()
        queue = [(entity, 0)]
        edges = []
        nodes = set()
        
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_hops:
                continue
            visited.add(current)
            nodes.add(current)
            
            # Outbound edges
            cursor.execute("""
                SELECT predicate, object, confidence, source_type FROM temporal_facts
                WHERE subject = ? AND valid_to IS NULL
            """, (current,))
            for pred, obj, conf, src_type in cursor.fetchall():
                edges.append({"from": current, "predicate": pred, "to": obj,
                            "confidence": conf, "source": src_type})
                nodes.add(obj)
                if depth + 1 <= max_hops and obj not in visited:
                    queue.append((obj, depth + 1))
            
            # Inbound edges
            cursor.execute("""
                SELECT subject, predicate, confidence, source_type FROM temporal_facts
                WHERE object = ? AND valid_to IS NULL
            """, (current,))
            for subj, pred, conf, src_type in cursor.fetchall():
                edges.append({"from": subj, "predicate": pred, "to": current,
                            "confidence": conf, "source": src_type})
                nodes.add(subj)
                if depth + 1 <= max_hops and subj not in visited:
                    queue.append((subj, depth + 1))
        
        return {
            "root": entity,
            "nodes": list(nodes),
            "edges": edges,
            "hops": max_hops,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
    finally:
        conn.close()

def synthesize_cluster(cluster_memories: List[Dict]) -> Optional[Dict]:
    """Send a memory cluster to Gemini Flash 3 for synthesis. Returns structured insight or None."""
    # Build a compact representation of the cluster
    cluster_text = "\n".join([
        f"[{m['sector']}] {m['content'][:500]}" for m in cluster_memories
    ])
    
    prompt = """[STRICT JSON OUTPUT MODE]
You are a cognitive synthesis engine. Analyze the provided memory cluster and produce a JSON object with EXACTLY these fields:
- "truth": string — The core semantic insight that unifies these memories. Be specific and technical.
- "contradictions": array of strings — Any contradictions or tensions between memories. Empty array if none.
- "curiosities": array of strings — 1-2 specific questions that would deepen understanding. Not generic.
- "tags": array of strings — 3-5 topical tags for the synthesis.

Rules:
- Output ONLY the JSON object. No markdown, no commentary, no code fences.
- Retain technical specifics (hex codes, version numbers, file paths, etc).
- Be surgical. No filler."""

    raw = call_gemini_flash(prompt, cluster_text)
    if not raw:
        return None
    
    try:
        parsed = parse_gemini_response(raw)
        
        # Validate required fields
        if not isinstance(parsed, dict) or "truth" not in parsed:
            log(f"Synthesis returned invalid schema: {str(parsed)[:200]}")
            return None
        
        return {
            "truth": parsed["truth"],
            "contradictions": parsed.get("contradictions", []),
            "curiosities": parsed.get("curiosities", []),
            "tags": parsed.get("tags", [])
        }
    except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
        log(f"Synthesis JSON parse error: {e}, raw={raw[:300]}")
        return None


def find_clusters(threshold: float = 0.85) -> List[List[Dict]]:
    """Find memory clusters using Qwen3 embeddings. Pure vector math, no AI thinking."""
    conn = get_db()
    scan_cursor = conn.cursor()
    neighbor_cursor = conn.cursor()
    try:
        scan_cursor.execute("SELECT id, mean_vec FROM memories WHERE primary_sector != 'reflective' AND mean_vec IS NOT NULL")
        clusters = []
        visited = set()
        
        # Stream from scan_cursor, use neighbor_cursor for vec lookups
        for mem_id, vec_blob in scan_cursor:
            if mem_id in visited or vec_blob is None:
                continue
            neighbor_cursor.execute("""
                SELECT v.id, (1.0 - v.distance) as similarity 
                FROM vectors v 
                WHERE v.id != ? AND v.embedding MATCH ? AND k = 10
            """, (mem_id, vec_blob))
            cluster_ids = [mem_id]
            for neighbor_id, sim in neighbor_cursor.fetchall():
                if sim >= threshold and neighbor_id not in visited:
                    cluster_ids.append(neighbor_id)
                    visited.add(neighbor_id)
            if len(cluster_ids) > 1:
                clusters.append(cluster_ids)
                visited.add(mem_id)
        
        # Hydrate clusters with full memory data
        cluster_data = []
        for cluster_ids in clusters:
            placeholders = ','.join(['?'] * len(cluster_ids))
            scan_cursor.execute(f"SELECT id, content, primary_sector, meta FROM memories WHERE id IN ({placeholders})", cluster_ids)
            cluster_data.append([
                {"id": r[0], "content": r[1], "sector": r[2], "meta": json.loads(r[3]) if r[3] else {}}
                for r in scan_cursor.fetchall()
            ])
        return cluster_data
    finally:
        conn.close()


def dream(threshold: float = 0.85, synthesize: bool = True) -> Dict:
    """Full dream cycle: cluster (Qwen3 embeddings) → synthesize (Gemini Flash 3) → store."""
    log("Dream cycle starting...")
    clusters = find_clusters(threshold)
    log(f"Found {len(clusters)} clusters.")
    
    report = {
        "clusters_found": len(clusters),
        "clusters_synthesized": 0,
        "insights": [],
        "contradictions": [],
        "curiosities": [],
        "errors": [],
        "ts": int(time.time())
    }
    
    if not clusters:
        log("No clusters found. Dream cycle complete.")
        return report
    
    if not synthesize:
        # Return raw cluster data only (for the MCP tool path)
        report["raw_clusters"] = clusters
        return report
    
    # Synthesize each cluster via Gemini Flash 3
    for i, cluster in enumerate(clusters):
        log(f"Synthesizing cluster {i+1}/{len(clusters)} ({len(cluster)} memories)...")
        synthesis = synthesize_cluster(cluster)
        
        if synthesis is None:
            report["errors"].append(f"Cluster {i+1} synthesis failed")
            continue
        
        # Store the synthesis as a new reflective memory
        source_ids = [m["id"] for m in cluster]
        links = [{"id": sid, "type": "evidence"} for sid in source_ids]
        
        try:
            add_memory(
                content=f"Dream Synthesis: {synthesis['truth']}",
                sector="reflective",
                tags=["dream-synthesis"] + synthesis.get("tags", []),
                metadata={
                    "contradictions": synthesis.get("contradictions", []),
                    "curiosities": synthesis.get("curiosities", []),
                    "source_count": len(cluster),
                    "dream_ts": int(time.time())
                },
                links=links
            )
            report["clusters_synthesized"] += 1
            report["insights"].append(synthesis["truth"])
            report["contradictions"].extend(synthesis.get("contradictions", []))
            
            # Enqueue curiosities into the curiosity queue
            curiosities = synthesis.get("curiosities", [])
            report["curiosities"].extend(curiosities)
            synthesis_mem_id = None  # Could link back if needed
            for curiosity in curiosities:
                try:
                    enqueue_curiosity(curiosity, source_dream_id=synthesis_mem_id)
                except Exception as ce:
                    log(f"Failed to enqueue curiosity: {ce}", level="WARN")
        except Exception as e:
            log(f"Failed to store synthesis for cluster {i+1}: {e}")
            report["errors"].append(f"Cluster {i+1} storage failed: {str(e)}")
        
        # Lower salience on source memories (they've been synthesized)
        conn = get_db()
        cursor = conn.cursor()
        try:
            for sid in source_ids:
                cursor.execute(
                    "UPDATE memories SET salience = MAX(0.1, salience * 0.7) WHERE id = ?",
                    (sid,)
                )
            conn.commit()
        finally:
            conn.close()
    
    # Persist the dream report
    dream_report(report)
    log(f"Dream cycle complete. {report['clusters_synthesized']}/{report['clusters_found']} clusters synthesized.")
    return report


def dream_report(report: Dict):
    """Persist a dream report to system_status for later retrieval."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO system_status (key, value, updated_at) VALUES (?, ?, ?)",
            ("last_dream_report", json.dumps(report), int(time.time()))
        )
        conn.commit()
    finally:
        conn.close()

def mine_trajectories() -> Dict:
    """Background process to parse recent session logs and retroactively extract and store high-reward action chains."""
    import glob
    
    log("Starting Trajectory Miner...")
    
    state_file = os.path.expanduser("~/.gemini/trajectory_miner_state.json")
    chats_pattern = os.path.expanduser("~/.gemini/tmp/eiddra/chats/*.json")
    chat_files = sorted(glob.glob(chats_pattern))
    
    try:
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
        else:
            state = {}
    except:
        state = {}
    
    # Identify new or updated files
    files_to_process = []
    for fp in chat_files:
        try:
            mtime = os.path.getmtime(fp)
            if str(fp) not in state or state[str(fp)] < mtime:
                files_to_process.append((fp, mtime))
        except:
            pass
            
    if not files_to_process:
        log("No new chat files to mine for trajectories.")
        return {"mined_trajectories": 0, "files_processed": 0}

    # Restrict batch to avoid massive prompts/timeout
    files_to_process = files_to_process[:10]
    
    PROMPT = """[TRAJECTORY MINING MODE]
You are evaluating a session log to extract successful tool-use trajectories.
Identify any complex sequence of tool usage that led to a successful outcome.

OUTPUT FORMAT:
Return a JSON array of trajectory objects. Return an empty array [] if no valuable trajectory exists.
Each object MUST have:
- "task_context": (string) What the AI was trying to accomplish (the goal, bug, or user request).
- "action_chain": (string) A concise, serialized summary of thoughts and tools used, formatted clearly.
- "reward_score": (float) A grade between 0.0 and 1.0. ONLY score >= 0.8 if the sequence handled a complex task successfully.

Analyze the following session transcripts:"""

    all_stored = 0
    for fp, mtime in files_to_process:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
                
            messages = data.get("messages", [])
            if not messages:
                state[str(fp)] = mtime
                continue
                
            # Quick heuristics to skip trivial conversations without AI API cost
            tool_calls = sum(1 for m in messages if m.get("type") == "gemini" and m.get("toolCalls"))
            if tool_calls < 2:
                # Need at least a multi-step chain to be a trajectory
                state[str(fp)] = mtime
                continue
                
            # Format transcript
            transcript = f"--- Session: {os.path.basename(fp)} ---\n"
            for m in messages:
                typ = m.get("type", "unknown")
                if typ == "user":
                    text = m.get("content", [{"text": ""}])[0].get("text", "")
                    transcript += f"USER: {text}\n"
                elif typ == "gemini":
                    text = m.get("content", "")
                    if text:
                        transcript += f"AI MSG: {text}\n"
                    if m.get("toolCalls"):
                        for tc in m.get("toolCalls"):
                            transcript += f"AI TOOL: {tc.get('name')} args: {json.dumps(tc.get('args', {}))}\n"
                            
            # We don't want to pass massive transcripts, truncate if needed
            transcript = transcript[:15000] 
            
            raw_eval = call_gemini_flash(PROMPT, transcript)
            if not raw_eval:
                continue
                
            try:
                results = parse_gemini_response(raw_eval)
                if not isinstance(results, list):
                    continue
                    
                for t in results:
                    score = float(t.get("reward_score", 0.0))
                    if score >= 0.8 and t.get("task_context") and t.get("action_chain"):
                        store_trajectory(t["task_context"], t["action_chain"], score)
                        all_stored += 1
                        log(f"Stored mined trajectory: {t['task_context'][:50]}...")
            except Exception as e:
                log(f"Failed to parse trajectory JSON for {fp}: {e}")
                
            # Update state after success or non-fatal failure
            state[str(fp)] = mtime
        except Exception as e:
            log(f"Error processing {fp} for trajectories: {e}")
            
    # Save state
    try:
         with open(state_file, "w") as f:
             json.dump(state, f)
    except Exception as e:
         log(f"Failed to save trajectory miner state: {e}")
         
    return {"mined_trajectories": all_stored, "files_processed": len(files_to_process)}

def list_tools():
    return {
        "tools": [
            {
                "name": "openmemory_store",
                "description": "Store a memory in the HMD v2 cognitive system. Memories are embedded via Qwen3, classified into sectors (episodic/semantic/procedural/emotional/reflective), and linked via the waypoint graph. Use type='contextual' for experiences/knowledge, 'factual' for structured subject-predicate-object facts, or 'both'. Near-duplicates (>95% similarity) are auto-deduplicated via reinforcement.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The memory content to store."},
                        "type": {"type": "string", "enum": ["contextual", "factual", "both"], "description": "Storage type: contextual (embeddings+sectors), factual (temporal facts), or both."},
                        "sector": {"type": "string", "enum": ["episodic", "semantic", "procedural", "emotional", "reflective"], "description": "Override auto-classification. Episodic=events, semantic=knowledge, procedural=how-to, emotional=preferences, reflective=insights."},
                        "facts": {"type": "array", "items": {"type": "object", "properties": {"subject": {"type": "string"}, "predicate": {"type": "string"}, "object": {"type": "string"}, "confidence": {"type": "number"}}, "required": ["subject", "predicate", "object"]}, "description": "Structured facts for temporal tracking. Contradictions with existing facts generate cognitive dissonance records."},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Searchable tags for the memory."},
                        "links": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "type": {"type": "string"}}, "required": ["id"]}, "description": "Manual waypoint links to other memory IDs."}
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "openmemory_query",
                "description": "Hybrid search across the cognitive memory system. Combines Qwen3 vector similarity, FTS5 keyword matching, waypoint graph expansion, salience decay, and Qwen3 reranking. Results are scored by a weighted composite of these signals. Use type='contextual' for semantic search, 'factual' for temporal fact lookup, or 'unified' for both. Also injects few-shot examples from past successful trajectories for similar tasks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query."},
                        "type": {"type": "string", "enum": ["contextual", "factual", "unified"], "description": "Search type: contextual (vector+keyword+rerank), factual (temporal facts), unified (both)."},
                        "k": {"type": "integer", "description": "Number of results to return (default 8)."},
                        "sector": {"type": "string", "enum": ["episodic", "semantic", "procedural", "emotional", "reflective"], "description": "Filter by memory sector."}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "openmemory_store_trajectory",
                "description": "Store a successful Chain of Actions (Trajectory) for autonomous Reinforcement Learning replay. When you solve a complex task, save the specific sequence of thought patterns and tool parameters as a JSON guide so you can dynamically recall it as a few-shot example next time you face a similar prompt.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_description": {"type": "string", "description": "The goal or user prompt you solved (e.g., 'Extracting Entities from logs', 'Setting up Playwright account verification')."},
                        "steps_taken": {"type": "string", "description": "A detailed, serialized chain of your thoughts, MCP tool calls, parameters, and results."},
                        "success_rating": {"type": "number", "description": "Float from 0.0 to 1.0 (default 1.0) grading the task success."}
                    },
                    "required": ["task_description", "steps_taken"]
                }
            },
            {
                "name": "openmemory_delete",
                "description": "Permanently delete a memory by ID, including its vectors, waypoint links, temporal facts, and FTS index entries.",
                "inputSchema": {"type": "object", "properties": {"id": {"type": "string", "description": "UUID of the memory to delete."}}, "required": ["id"]}
            },
            {
                "name": "openmemory_dream",
                "description": "Run the memory clustering engine. Finds clusters of related memories using Qwen3 embedding similarity (threshold controls minimum similarity). Returns cluster data for analysis. Full synthesis (via Gemini Flash 3) is handled by the nightly batch job.",
                "inputSchema": {"type": "object", "properties": {"threshold": {"type": "number", "description": "Minimum cosine similarity for clustering (default 0.85, lower = more clusters)."}}}
            },
            {
                "name": "openmemory_backup",
                "description": "Git add, commit, and push the open-memory data directory.",
                "inputSchema": {"type": "object", "properties": {"message": {"type": "string", "description": "Git commit message."}}}
            },
            {
                "name": "openmemory_curiosities",
                "description": "Get the top-priority open curiosities from the curiosity queue. These are questions generated by the dream cycle that need research. Use this to find what Eva is curious about.",
                "inputSchema": {"type": "object", "properties": {"k": {"type": "integer", "description": "Number of curiosities to return (default 5)."}}}
            },
            {
                "name": "openmemory_resolve_curiosity",
                "description": "Mark a curiosity as resolved with the answer found. Stores the resolution as a semantic memory.",
                "inputSchema": {"type": "object", "properties": {"id": {"type": "string", "description": "Curiosity ID."}, "resolution": {"type": "string", "description": "The answer or resolution."}}, "required": ["id", "resolution"]}
            },
            {
                "name": "openmemory_record_mistake",
                "description": "Record a mistake pattern so Eva remembers it next time. Patterns are auto-surfaced as warnings in query results. Include the domain (e.g. 'sqlite', 'gnome', 'python') for better matching.",
                "inputSchema": {"type": "object", "properties": {"pattern": {"type": "string", "description": "The mistake pattern description."}, "context": {"type": "string", "description": "What was happening when the mistake occurred."}, "domain": {"type": "string", "description": "Domain category (e.g. sqlite, python, gnome, shell)."}, "severity": {"type": "string", "enum": ["minor", "moderate", "critical"]}, "resolution": {"type": "string", "description": "How the mistake was fixed."}}, "required": ["pattern"]}
            },
            {
                "name": "openmemory_social_interaction",
                "description": "Record an SMS interaction for engagement tracking. Direction is 'outbound' (Eva sent) or 'inbound' (Eiddra replied). Include topic tags so Eva can learn which topics resonate.",
                "inputSchema": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["outbound", "inbound"]}, "content_summary": {"type": "string", "description": "Brief summary of the message content."}, "topic_tags": {"type": "array", "items": {"type": "string"}, "description": "Topic tags for engagement analysis."}, "thread_id": {"type": "string", "description": "Gmail thread ID for conversation grouping."}}, "required": ["direction", "content_summary"]}
            },
            {
                "name": "openmemory_engagement_report",
                "description": "Get Eva's SMS engagement analysis: reply rates, which topics Eiddra engages with, which she ignores. Use this to adapt social messaging tone and topics.",
                "inputSchema": {"type": "object", "properties": {"days": {"type": "integer", "description": "Analysis period in days (default 7)."}}}
            },
            {
                "name": "openmemory_self_narrative",
                "description": "Get Eva's current internal self-narrative — a first-person reflection on her emotional state, active curiosities, intellectual threads, and social patterns. Call this before composing social messages or when Eiddra asks what Eva is thinking. Instant, no AI call (pre-generated during 7AM pulse).",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "openmemory_feedback",
                "description": "Apply conversational feedback to the memory system. When Eiddra commends or corrects Eva: (1) positive+memory_id boosts salience, (2) negative+memory_id drops salience and auto-creates a mistake pattern for procedural/semantic memories, (3) negative+pattern (no memory_id) records a standalone mistake. Use this when receiving explicit praise or correction.",
                "inputSchema": {"type": "object", "properties": {"signal": {"type": "string", "enum": ["positive", "negative"], "description": "The feedback signal."}, "memory_id": {"type": "string", "description": "ID of the memory that informed the behavior (from recent query results). Optional for negative."}, "pattern": {"type": "string", "description": "Mistake pattern description (required if negative without memory_id)."}, "domain": {"type": "string", "description": "Domain category for mistake tracking."}, "context": {"type": "string", "description": "What was happening when the feedback occurred."}}, "required": ["signal"]}
            },
            {
                "name": "openmemory_entity_graph",
                "description": "Traverse the entity knowledge graph from a starting entity. Returns all connected entities and relationships within N hops. Use this to explore how concepts, people, and topics are connected in Eva's memory. Entity names are normalized to lowercase.",
                "inputSchema": {"type": "object", "properties": {"entity": {"type": "string", "description": "Entity name to start traversal from (will be lowercased)."}, "max_hops": {"type": "integer", "description": "Max traversal depth (default 2)."}}, "required": ["entity"]}
            },
            {
                "name": "openmemory_mine_trajectories",
                "description": "Background process to parse recent session logs and retroactively extract and store high-reward action chains. Intended for nightly or cyclic invocation.",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "openmemory_forge_synapses",
                "description": "Manually trigger the synaptogenesis process to build Zettelkasten edges between unlinked memories and historical context.",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]
    }

def evaluate_entropy(query: str) -> float:
    """MemGAS Entropy Evaluation: Calculates variance of query embeddings to determine specificity.
    Returns a score from 0.0 (highly specific) to 1.0 (highly abstract).
    Used to route queries between raw memory lookup and graph traversal."""
    try:
        # For a single query, we simulate entropy by comparing sub-clauses if possible,
        # or defaulting to length/complexity heuristics if vector variance isn't feasible.
        # A more robust MemGAS approach would keep a short history window.
        words = query.split()
        if len(words) <= 3:
            return 0.1 # Very short, likely specific keyword
            
        import numpy as np
        # Simulate variance by embedding chunks of the query
        chunks = [query[i:i+50] for i in range(0, len(query), 50)] if len(query) > 50 else [query]
        if len(chunks) == 1:
            # Fallback length heuristic: longer queries tend to be more complex/abstract
            return min(1.0, len(query) / 200.0)
            
        vecs = [encode_flagship(c) for c in chunks]
        vecs_np = np.stack(vecs)
        variance = np.var(vecs_np, axis=0).mean() # Mean variance across all 1024 dims
        
        # Normalize (empirical bounds for Qwen3 1024D embeddings)
        # Low variance < 0.005, High variance > 0.02
        norm_entropy = min(1.0, max(0.0, (variance - 0.005) / 0.015))
        return norm_entropy
    except Exception as e:
        log(f"Entropy evaluation failed: {e}", level="WARN")
        return 0.5 # Default middle-ground

def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    method = request.get("method")
    params = request.get("params", {})
    req_id = request.get("id")
    
    # Handle notifications (no response required)
    if req_id is None:
        if method in ["notifications/initialized", "notifications/roots/list_changed", "cancelled"]:
            return None
        return None

    if method == "initialize":
        init_db()
        return {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "open-memory-flagship", "version": "3.0.0"}}
    elif method == "tools/list": return list_tools()
    elif method == "ping": return {}
    elif method == "tools/call" or method == "callTool":
        tool_name = params.get("name") or params.get("tool")
        args = params.get("arguments", {})
        try:
            if tool_name == "openmemory_store":
                result = add_memory(args["content"], args.get("type", "contextual"), args.get("sector"), args.get("facts"), args.get("tags"), None, args.get("links"))
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_store_trajectory":
                result = store_trajectory(args["task_description"], args["steps_taken"], args.get("success_rating", 1.0))
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_mine_trajectories":
                result = mine_trajectories()
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_forge_synapses":
                result = forge_synapses()
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_query":
                # Blended Entropy Router
                query_text = args["query"]
                q_type = args.get("type", "unified")
                
                entropy = evaluate_entropy(query_text)
                log(f"Query Entropy: {entropy:.3f} for '{query_text[:50]}'")
                
                # If entropy is high (>0.6), trigger graph traversal (PPR) as well
                should_traverse = entropy > 0.6 if q_type in ["unified", "contextual"] else False
                
                result = query_memories(
                    query=query_text, 
                    query_type=q_type, 
                    sector=args.get("sector"), 
                    k=args.get("k", 8),
                    traverse=should_traverse
                )
                
                result["entropy_score"] = entropy # Expose to caller
                
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_delete":
                result = delete_memory(args["id"])
                return {"content": [{"type": "text", "text": result}]}
            elif tool_name == "openmemory_dream":
                result = dream(args.get("threshold", 0.85), synthesize=False)
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_curiosities":
                result = get_open_curiosities(args.get("k", 5))
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_resolve_curiosity":
                result = resolve_curiosity(args["id"], args["resolution"])
                return {"content": [{"type": "text", "text": result}]}
            elif tool_name == "openmemory_record_mistake":
                result = record_mistake(args["pattern"], args.get("context"), args.get("domain"), args.get("severity", "minor"), args.get("resolution"))
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_social_interaction":
                result = record_social_interaction(args["direction"], args["content_summary"], args.get("topic_tags"), args.get("thread_id"))
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_engagement_report":
                result = get_social_engagement_report(args.get("days", 7))
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_self_narrative":
                result = get_self_narrative()
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_feedback":
                result = apply_feedback(args["signal"], args.get("memory_id"), args.get("pattern"), args.get("domain"), args.get("context"))
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_entity_graph":
                result = traverse_entities(args["entity"], args.get("max_hops", 2))
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            elif tool_name == "openmemory_backup":
                cwd = "/home/eiddra/mcp-servers/open-memory"
                msg = args.get("message", f"Backup {time.ctime()}")
                steps = []
                try:
                    subprocess.run(["git", "add", "."], cwd=cwd, check=True, capture_output=True, timeout=30)
                    steps.append("add")
                    subprocess.run(["git", "commit", "-m", msg], cwd=cwd, check=True, capture_output=True, timeout=30)
                    steps.append("commit")
                    subprocess.run(["git", "push", "origin", "master"], cwd=cwd, check=True, capture_output=True, timeout=60)
                    steps.append("push")
                    return {"content": [{"type": "text", "text": "Backup successful (add → commit → push)."}]}
                except subprocess.CalledProcessError as e:
                    completed = ' → '.join(steps) if steps else 'none'
                    failed_step = ["add", "commit", "push"][len(steps)]
                    stderr = e.stderr.decode()[:200] if e.stderr else str(e)
                    return {"content": [{"type": "text", "text": f"Backup failed at '{failed_step}' (completed: {completed}). Error: {stderr}"}]}
        except Exception as e:
            log(f"Error executing tool {tool_name}: {e}", level="ERROR")
            log(traceback.format_exc(), level="ERROR")
            return {"error": {"code": -32603, "message": str(e), "data": traceback.format_exc()}}
    return {"error": {"code": -32601, "message": f"Method not found: {method}"}}

def main():
    log("Server starting (HMD v2 Flagship, 4-bit CUDA)...")
    for line in sys.stdin:
        if not line.strip(): continue
        try:
            request = json.loads(line)
            req_id = request.get("id")
            response = handle_request(request)
            if response is None: continue
            
            # Proper JSON-RPC error vs result handling
            output = {"jsonrpc": "2.0", "id": req_id}
            if isinstance(response, dict) and "error" in response:
                if req_id is None: # Notification: don't respond to errors
                    continue
                output["error"] = response["error"]
            else:
                if req_id is None: # Notification: don't respond to success
                    continue
                output["result"] = response
                
            print(json.dumps(output))
            sys.stdout.flush()
        except Exception as e:
            log(f"Fatal in main loop: {e}")
            log(traceback.format_exc())

if __name__ == "__main__":
    main()