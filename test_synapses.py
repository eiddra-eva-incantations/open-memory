from mcp_server import traverse_synapses, init_db
import json

init_db()
res = traverse_synapses("What is the core architecture?", k=3)
for r in res:
    print(r["id"])
    print(f"PPR: {r['ppr_score']:.3f} | Sim: {r['similarity']:.3f}")
    print(r["content"][:200])
    print("-" * 40)
