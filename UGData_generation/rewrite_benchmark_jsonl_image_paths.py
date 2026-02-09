r"""
Rewrite image paths in benchmark JSONL files to a normalized prefix.

Goal:
  Replace absolute image paths in benchmark JSONL files to benchmark/benchmark_images/

Example transformations:
  - "/home/xingtong/UrbanKG/data/geo/SR/osm_data/beijing_paris_tokyo/643285156523139.jpg"
    -> "benchmark/benchmark_images/643285156523139.jpg"
  
  - For files with "candidates" list (retrieval tasks):
    Rewrites all paths in "candidates" and "candidate_images" lists
  
  - For files with "images" field (geolocation tasks):
    Rewrites the "images" value

This script:
  - Scans for JSONL files in benchmark directories
  - Handles different field structures (candidates, images, ground_truth)
  - Preserves all other fields as-is
  - Supports dry-run and optional backup files

Usage examples:
  python ./UGData_generation/rewrite_benchmark_jsonl_image_paths.py \
    --dir ./benchmark \
    --prefix benchmark/benchmark_images/ \
    --backup

  # single file
  python ./UGData_generation/rewrite_benchmark_jsonl_image_paths.py \
    --file ./benchmark/retrieval/beijing_with_graph.jsonl \
    --prefix benchmark/benchmark_images/ \
    --backup
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Tuple


def _rewrite_image_path(path: str, prefix: str) -> str:
    """Rewrite a single image path to use the new prefix."""
    if not path or not isinstance(path, str):
        return path
    filename = os.path.basename(path)
    return f"{prefix}{filename}"


def _rewrite_images_value(images: Any, prefix: str) -> Tuple[Any, int]:
    """Return (new_images, num_changed). Handles both list[str] and str entries."""
    if isinstance(images, str) and images.strip():
        # Single string path
        new_item = _rewrite_image_path(images, prefix)
        if new_item != images:
            return new_item, 1
        return images, 0
    
    if not isinstance(images, list):
        return images, 0

    changed = 0
    new_list = []
    for item in images:
        if isinstance(item, str) and item.strip():
            new_item = _rewrite_image_path(item, prefix)
            if new_item != item:
                changed += 1
            new_list.append(new_item)
        else:
            new_list.append(item)
    return new_list, changed


def rewrite_jsonl_file(path: Path, prefix: str, *, dry_run: bool, backup: bool) -> Tuple[int, int, int]:
    """
    Rewrite one JSONL file in-place.
    Returns: (lines_total, lines_changed, images_changed)
    """
    lines_total = 0
    lines_changed = 0
    images_changed = 0

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
        
        # Handle "images" field (geolocation tasks)
        if isinstance(obj, dict) and "images" in obj:
            new_images, n = _rewrite_images_value(obj.get("images"), prefix)
            if n:
                obj["images"] = new_images
                changed_this_line = True
                images_changed += n
        
        # Handle "candidates" field (retrieval tasks)
        if isinstance(obj, dict) and "candidates" in obj:
            new_candidates, n = _rewrite_images_value(obj.get("candidates"), prefix)
            if n:
                obj["candidates"] = new_candidates
                changed_this_line = True
                images_changed += n
        
        # Handle "candidate_images" field (retrieval tasks)
        if isinstance(obj, dict) and "candidate_images" in obj:
            new_candidate_images, n = _rewrite_images_value(obj.get("candidate_images"), prefix)
            if n:
                obj["candidate_images"] = new_candidate_images
                changed_this_line = True
                images_changed += n
        
        # Handle "ground_truth" field if it's an image path
        if isinstance(obj, dict) and "ground_truth" in obj:
            ground_truth = obj.get("ground_truth")
            if isinstance(ground_truth, str) and ground_truth.strip() and ground_truth.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                new_gt = _rewrite_image_path(ground_truth, prefix)
                if new_gt != ground_truth:
                    obj["ground_truth"] = new_gt
                    changed_this_line = True
                    images_changed += 1
        
        if changed_this_line:
            lines_changed += 1
        
        out_lines.append(json.dumps(obj, ensure_ascii=False))

    if dry_run:
        return lines_total, lines_changed, images_changed

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            bak.write_text("\n".join(raw_lines) + ("\n" if raw_lines else ""), encoding="utf-8")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    tmp.replace(path)

    return lines_total, lines_changed, images_changed


def iter_benchmark_jsonl_files(dir_path: Path) -> Iterable[Path]:
    """Find all JSONL files recursively in benchmark directory."""
    for p in dir_path.rglob("*.jsonl"):
        if p.is_file():
            yield p


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rewrite JSONL image paths in benchmark files to a normalized prefix."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=Path, help="Path to a single .jsonl file")
    g.add_argument("--dir", type=Path, help="Directory to scan recursively for .jsonl files")

    p.add_argument(
        "--prefix",
        default="benchmark/benchmark_images/",
        help="New prefix for images paths (default: benchmark/benchmark_images/)",
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
    total_images_changed = 0

    for f in files:
        total_files += 1
        lines_total, lines_changed, images_changed = rewrite_jsonl_file(
            f, prefix, dry_run=args.dry_run, backup=args.backup
        )
        total_lines += lines_total
        total_lines_changed += lines_changed
        total_images_changed += images_changed
        print(
            f"{f}: lines={lines_total}, lines_changed={lines_changed}, images_changed={images_changed}"
        )

    print(
        f"\nDone. files={total_files}, lines={total_lines}, lines_changed={total_lines_changed}, images_changed={total_images_changed}"
        + (" (dry-run)" if args.dry_run else "")
    )


if __name__ == "__main__":
    main()

