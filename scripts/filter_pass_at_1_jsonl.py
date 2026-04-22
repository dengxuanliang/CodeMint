#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


OUTPUT_SUFFIX = ".pass_at_1_0.jsonl"


@dataclass(frozen=True, slots=True)
class FilterStats:
    input_path: Path
    output_path: Path
    total_rows: int
    kept_rows: int


def filter_jsonl_file(input_path: Path, output_path: Path) -> FilterStats:
    total_rows = 0
    kept_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {input_path}:{line_number}: {error.msg}") from error
            if row.get("pass_at_1") == 0:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_rows += 1

    return FilterStats(
        input_path=input_path,
        output_path=output_path,
        total_rows=total_rows,
        kept_rows=kept_rows,
    )


def filter_directory(directory: Path, *, in_place: bool = False) -> list[FilterStats]:
    stats: list[FilterStats] = []
    for input_path in _iter_source_files(directory):
        output_path = input_path if in_place else input_path.with_suffix("").with_name(input_path.stem + OUTPUT_SUFFIX)
        if in_place:
            temporary_path = input_path.with_suffix(input_path.suffix + ".tmp")
            stat = filter_jsonl_file(input_path, temporary_path)
            temporary_path.replace(input_path)
            stat = FilterStats(
                input_path=stat.input_path,
                output_path=input_path,
                total_rows=stat.total_rows,
                kept_rows=stat.kept_rows,
            )
        else:
            stat = filter_jsonl_file(input_path, output_path)
        stats.append(stat)
    return stats


def _iter_source_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.jsonl")):
        if path.name.endswith(OUTPUT_SUFFIX):
            continue
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter JSONL files under a directory, keeping only rows where pass_at_1 is numeric 0."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="real_log_files",
        type=Path,
        help="Directory containing JSONL files. Defaults to real_log_files.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each source JSONL file instead of writing *.pass_at_1_0.jsonl outputs.",
    )
    args = parser.parse_args()

    stats = filter_directory(args.directory, in_place=args.in_place)
    for stat in stats:
        print(f"{stat.input_path} -> {stat.output_path}: kept {stat.kept_rows}/{stat.total_rows}")


if __name__ == "__main__":
    main()
