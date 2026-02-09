r"""
Rewrite `images` paths in JSONL files to a normalized prefix.

Goal (per user request):
  "{DATA_ROOT}/.../images/243598200898585.jpg"
    -> "/mydata/stage1_images/243598200898585.jpg"

This script:
  - Scans one JSONL file or a directory of JSONL files
  - For each JSON line (one JSON object per line), rewrites `images` entries
  - Preserves all other fields as-is
  - Supports dry-run and optional backup files

Usage examples:
  python ./UGData_generation/rewrite_stage1_jsonl_image_paths.py \
    --dir ./mydata/stage1_image_text_pairs_complete \
    --prefix /mydata/stage1_images/ \
    --backup

  # single file
  python ./UGData_generation/rewrite_stage1_jsonl_image_paths.py \
    --file ./mydata/stage1_image_text_pairs_complete/mapillary_results_singapore_captions_swift_format_cleaned_add_stage1.jsonl \
    --prefix /mydata/stage1_images/ \
    --backup
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Tuple


def _rewrite_images_value(images: Any, prefix: str) -> Tuple[Any, int]:
    """Return (new_images, num_changed). Only rewrites list[str] entries."""
    if not isinstance(images, list):
        return images, 0

    changed = 0
    new_list = []
    for item in images:
        if isinstance(item, str) and item.strip():
            filename = os.path.basename(item)
            new_item = f"{prefix}{filename}"
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
        if isinstance(obj, dict) and "images" in obj:
            new_images, n = _rewrite_images_value(obj.get("images"), prefix)
            if n:
                obj["images"] = new_images
                lines_changed += 1
                images_changed += n
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


def iter_jsonl_files(dir_path: Path) -> Iterable[Path]:
    yield from sorted(p for p in dir_path.rglob("*.jsonl") if p.is_file())


def main() -> None:
    p = argparse.ArgumentParser(description="Rewrite JSONL `images` paths to a normalized prefix.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=Path, help="Path to a single .jsonl file")
    g.add_argument("--dir", type=Path, help="Directory to scan recursively for .jsonl files")

    p.add_argument(
        "--prefix",
        default="/mydata/stage1_images/",
        help="New prefix for images paths (default: /mydata/stage1_images/)",
    )
    p.add_argument("--dry-run", action="store_true", help="Show what would change, but do not write files")
    p.add_argument("--backup", action="store_true", help="Create a one-time .bak backup next to each file")

    args = p.parse_args()
    prefix: str = args.prefix
    if not prefix.endswith("/"):
        prefix += "/"

    files = [args.file] if args.file else list(iter_jsonl_files(args.dir))
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


