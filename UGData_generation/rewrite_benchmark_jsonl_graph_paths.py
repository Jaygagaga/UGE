r"""
Rewrite graph paths in benchmark JSONL files to a normalized prefix.

Goal:
  Replace absolute graph paths in benchmark JSONL files to benchmark/benchmark_graphs/

Example transformations:
  - "/home/xingtong/ms_swift/mydata/enhanced_image_data_with_paths_and_captions_beijing/subgraphs_with_paths/extended_180323017274361_subgraph_180323017274361.pkl"
    -> "benchmark/benchmark_graphs/extended_180323017274361_subgraph_180323017274361.pkl"
  
  - For files with "graphs" field:
    Rewrites the single graph path
  
  - For files with "candidate_graphs" field:
    Rewrites all paths in the candidate_graphs list

This script:
  - Scans for JSONL files in benchmark directories
  - Handles different field structures (graphs, candidate_graphs)
  - Preserves all other fields as-is
  - Supports dry-run and optional backup files

Usage examples:
  python ./UGData_generation/rewrite_benchmark_jsonl_graph_paths.py \
    --dir ./benchmark \
    --prefix benchmark/benchmark_graphs/ \
    --backup

  # single file
  python ./UGData_generation/rewrite_benchmark_jsonl_graph_paths.py \
    --file ./benchmark/retrieval/beijing_with_graph.jsonl \
    --prefix benchmark/benchmark_graphs/ \
    --backup
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Tuple


def _rewrite_graph_path(path: str, prefix: str) -> str:
    """Rewrite a single graph path to use the new prefix."""
    if not path or not isinstance(path, str):
        return path
    filename = os.path.basename(path)
    return f"{prefix}{filename}"


def _rewrite_graphs_value(graphs: Any, prefix: str) -> Tuple[Any, int]:
    """Return (new_graphs, num_changed). Handles both list[str] and str entries."""
    if isinstance(graphs, str) and graphs.strip():
        # Single string path
        new_item = _rewrite_graph_path(graphs, prefix)
        if new_item != graphs:
            return new_item, 1
        return graphs, 0
    
    if not isinstance(graphs, list):
        return graphs, 0

    changed = 0
    new_list = []
    for item in graphs:
        if isinstance(item, str) and item.strip():
            new_item = _rewrite_graph_path(item, prefix)
            if new_item != item:
                changed += 1
            new_list.append(new_item)
        else:
            new_list.append(item)
    return new_list, changed


def rewrite_jsonl_file(path: Path, prefix: str, *, dry_run: bool, backup: bool) -> Tuple[int, int, int]:
    """
    Rewrite one JSONL file in-place.
    Returns: (lines_total, lines_changed, graphs_changed)
    """
    lines_total = 0
    lines_changed = 0
    graphs_changed = 0

    # Read all lines first so we can write atomically.
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []

    for raw in raw_lines:
        lines_total += 1
        if not raw.strip():
            out_lines.append(raw)
            continue

        obj = json.loads(raw)
        changed_this_line = False
        
        # Handle "graphs" field (single graph path)
        if isinstance(obj, dict) and "graphs" in obj:
            graphs = obj.get("graphs")
            # Only rewrite if it's a non-empty string that looks like a graph path
            if isinstance(graphs, str) and graphs.strip() and graphs.endswith(('.pkl', '.pickle')):
                new_graphs = _rewrite_graph_path(graphs, prefix)
                if new_graphs != graphs:
                    obj["graphs"] = new_graphs
                    changed_this_line = True
                    graphs_changed += 1
        
        # Handle "candidate_graphs" field (list of graph paths)
        if isinstance(obj, dict) and "candidate_graphs" in obj:
            new_candidate_graphs, n = _rewrite_graphs_value(obj.get("candidate_graphs"), prefix)
            if n:
                obj["candidate_graphs"] = new_candidate_graphs
                changed_this_line = True
                graphs_changed += n
        
        if changed_this_line:
            lines_changed += 1
        
        out_lines.append(json.dumps(obj, ensure_ascii=False))

    if dry_run:
        return lines_total, lines_changed, graphs_changed

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            bak.write_text("\n".join(raw_lines) + ("\n" if raw_lines else ""), encoding="utf-8")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    tmp.replace(path)

    return lines_total, lines_changed, graphs_changed


def iter_benchmark_jsonl_files(dir_path: Path) -> Iterable[Path]:
    """Find all JSONL files recursively in benchmark directory."""
    for p in dir_path.rglob("*.jsonl"):
        if p.is_file():
            yield p


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rewrite JSONL graph paths in benchmark files to a normalized prefix."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=Path, help="Path to a single .jsonl file")
    g.add_argument("--dir", type=Path, help="Directory to scan recursively for .jsonl files")

    p.add_argument(
        "--prefix",
        default="benchmark/benchmark_graphs/",
        help="New prefix for graph paths (default: benchmark/benchmark_graphs/)",
    )
    p.add_argument("--dry-run", action="store_true", help="Show what would change, but do not write files")
    p.add_argument("--backup", action="store_true", help="Create a one-time .bak backup next to each file")

    args = p.parse_args()
    prefix: str = args.prefix
    if not prefix.endswith("/"):
        prefix += "/"

    if args.file:
        files = [args.file]
    else:
        files = sorted(iter_benchmark_jsonl_files(args.dir))
    
    if not files:
        raise SystemExit("No .jsonl files found.")

    total_files = 0
    total_lines = 0
    total_lines_changed = 0
    total_graphs_changed = 0

    for f in files:
        total_files += 1
        lines_total, lines_changed, graphs_changed = rewrite_jsonl_file(
            f, prefix, dry_run=args.dry_run, backup=args.backup
        )
        total_lines += lines_total
        total_lines_changed += lines_changed
        total_graphs_changed += graphs_changed
        print(
            f"{f}: lines={lines_total}, lines_changed={lines_changed}, graphs_changed={graphs_changed}"
        )

    print(
        f"\nDone. files={total_files}, lines={total_lines}, lines_changed={total_lines_changed}, graphs_changed={total_graphs_changed}"
        + (" (dry-run)" if args.dry_run else "")
    )


if __name__ == "__main__":
    main()

