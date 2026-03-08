#!/usr/bin/env python3
"""Standalone dream cycle runner. Call directly from shell scripts — no MCP overhead.

Usage:
    python3 dream_cycle.py [--threshold 0.6] [--no-synthesize] [--consolidate] [--quiet]

Exit codes:
    0 = success
    1 = dream failed
"""
import sys
import os
import subprocess
import json
import argparse

# Add parent to path so we can import mcp_server functions directly
sys.path.insert(0, '/home/eiddra/mcp-servers/open-memory')

from mcp_server import (dream, init_db, find_consolidation_candidates, consolidate_memories,
                        generate_self_narrative, extract_entities_batch, forge_synapses)

def main():
    parser = argparse.ArgumentParser(description="Run an open-memory dream cycle")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Similarity threshold for clustering (default: 0.6)")
    parser.add_argument("--no-synthesize", action="store_true",
                        help="Skip synthesis, only return clusters")
    parser.add_argument("--consolidate", action="store_true",
                        help="Run memory consolidation after dream cycle")
    parser.add_argument("--update-narrative", action="store_true",
                        help="Regenerate Eva's self-narrative after dream/consolidation")
    parser.add_argument("--extract-entities", action="store_true",
                        help="Batch extract entities from new memories into the knowledge graph")
    parser.add_argument("--forge-synapses", action="store_true",
                        help="Trigger synaptogenesis to build Zettelkasten edges between memories")
    parser.add_argument("--quiet", action="store_true",
                        help="Only output final JSON report")
    args = parser.parse_args()

    if not args.quiet:
        print(f"Dream cycle starting (threshold={args.threshold}, synthesize={not args.no_synthesize})...")

    # Pre-flight: kill orphaned MCP server processes to free VRAM
    cleanup_script = os.path.expanduser("~/.local/bin/cleanup-mcp.sh")
    if os.path.exists(cleanup_script):
        try:
            subprocess.run([cleanup_script], timeout=10, capture_output=True)
            if not args.quiet:
                print("Pre-flight MCP cleanup done.")
        except Exception:
            pass

    # Ensure DB schema is current
    init_db()

    report = dream(threshold=args.threshold, synthesize=not args.no_synthesize)

    if not args.quiet:
        print(f"\n--- Dream Report ---")
        print(f"Clusters found: {report.get('clusters_found', 0)}")
        print(f"Clusters synthesized: {report.get('clusters_synthesized', 0)}")
        if report.get('insights'):
            print(f"\nInsights:")
            for i, insight in enumerate(report['insights'], 1):
                print(f"  {i}. {insight[:200]}")
        if report.get('contradictions'):
            print(f"\nContradictions:")
            for c in report['contradictions']:
                print(f"  - {c[:200]}")
        if report.get('curiosities'):
            print(f"\nCuriosities (enqueued for research):")
            for c in report['curiosities']:
                print(f"  ? {c[:200]}")
        if report.get('errors'):
            print(f"\nErrors:")
            for e in report['errors']:
                print(f"  ! {e}")
        print(f"---")

    # Memory consolidation (optional, typically monthly)
    if args.consolidate:
        if not args.quiet:
            print(f"\n--- Memory Consolidation ---")
        candidates = find_consolidation_candidates()
        if candidates:
            if not args.quiet:
                print(f"Found {len(candidates)} consolidation candidates ({sum(len(c) for c in candidates)} memories)")
            consol_report = consolidate_memories(candidates)
            report["consolidation"] = consol_report
            if not args.quiet:
                print(f"Consolidated: {consol_report['consolidated']} clusters, {consol_report['memories_compressed']} memories compressed")
                if consol_report.get('errors'):
                    for e in consol_report['errors']:
                        print(f"  ! {e}")
        else:
            if not args.quiet:
                print("No consolidation candidates found.")
            report["consolidation"] = {"consolidated": 0, "memories_compressed": 0}
        if not args.quiet:
            print(f"---")

    # Self-narrative update (typically called by 7AM pulse or standalone)
    if args.update_narrative:
        if not args.quiet:
            print(f"\n--- Self-Narrative Generation ---")
        try:
            narrative = generate_self_narrative()
            report["narrative_updated"] = True
            report["narrative_length"] = len(narrative.get("text", ""))
            if not args.quiet:
                print(f"Narrative generated ({report['narrative_length']} chars)")
                print(f"Preview: {narrative['text'][:200]}...")
        except Exception as e:
            report["narrative_updated"] = False
            if not args.quiet:
                print(f"Narrative generation failed: {e}")
        if not args.quiet:
            print(f"---")

    # Entity extraction (batch)
    if args.extract_entities:
        if not args.quiet:
            print(f"\n--- Entity Extraction ---")
        try:
            entity_report = extract_entities_batch()
            report["entity_extraction"] = entity_report
            if not args.quiet:
                print(f"Processed {entity_report['processed']} memories, "
                      f"{entity_report['entities_added']} triples added")
                if entity_report.get('errors'):
                    for e in entity_report['errors']:
                        print(f"  ! {e}")
        except Exception as e:
            report["entity_extraction"] = {"error": str(e)}
            if not args.quiet:
                print(f"Entity extraction failed: {e}")
        if not args.quiet:
            print(f"---")

    # Synaptogenesis
    if args.forge_synapses:
        if not args.quiet:
            print(f"\n--- Synaptogenesis (Forging Edges) ---")
        try:
            synapse_report = forge_synapses()
            report["synaptogenesis"] = synapse_report
            if not args.quiet:
                print(f"Forged {synapse_report.get('edges_created', 0)} new associative edges")
                if synapse_report.get('errors'):
                    for e in synapse_report['errors']:
                        print(f"  ! {e}")
        except Exception as e:
            report["synaptogenesis"] = {"error": str(e)}
            if not args.quiet:
                print(f"Synaptogenesis failed: {e}")
        if not args.quiet:
            print(f"---")

    # Always output machine-readable JSON to stdout for piping
    print(json.dumps(report, indent=2))

    # Exit with error if all clusters failed
    if report.get('clusters_found', 0) > 0 and report.get('clusters_synthesized', 0) == 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
