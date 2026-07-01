#!/usr/bin/env python3
"""Stdlib-only offline self-check for the committed VNS database."""

from __future__ import annotations

import argparse
from pathlib import Path

from openhouse_utils import (
    cosine_similarity,
    load_bovw_histograms,
    load_database_metadata,
    resolve_db_source_path,
)


REPO = Path(__file__).resolve().parents[1]
DB_PATH = REPO / "simulation/database/qau_campus.vnsdb"
IMAGES_ROOT = REPO / "simulation/database/images_real"


def build_report() -> str:
    metadata = load_database_metadata(DB_PATH)
    histograms = load_bovw_histograms(DB_PATH, metadata)

    entries: list[tuple[str, Path, list[float]]] = []
    for entry_meta, histogram in histograms:
        entry_id = str(entry_meta.get("id", ""))
        source_path = resolve_db_source_path(DB_PATH, str(entry_meta.get("source_path", "")))
        if not entry_id or not histogram:
            continue
        if not source_path.is_file():
            continue
        if IMAGES_ROOT not in source_path.parents:
            continue
        entries.append((entry_id, source_path, histogram))

    if not entries:
        raise ValueError("No database entries resolved under simulation/database/images_real/.")

    top1_hits = 0
    top5_hits = 0
    example_lines: list[str] = []

    for query_id, _, query_hist in entries:
        scored = [
            (candidate_id, cosine_similarity(query_hist, candidate_hist))
            for candidate_id, _, candidate_hist in entries
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        top5 = scored[:5]
        top_ids = [candidate_id for candidate_id, _ in top5]
        if top_ids and top_ids[0] == query_id:
            top1_hits += 1
        if query_id in top_ids:
            top5_hits += 1
        if len(example_lines) < 10:
            summary = ", ".join(f"{candidate}:{score:.4f}" for candidate, score in top5)
            example_lines.append(f"{query_id} -> {summary}")

    total = len(entries)
    top1_pct = top1_hits * 100.0 / total
    top5_pct = top5_hits * 100.0 / total

    lines = [
        "Offline VNS Database Self-Check",
        "===============================",
        f"Images root: {IMAGES_ROOT}",
        f"Database: {DB_PATH}",
        f"Entries evaluated: {total}",
        (
            "Method: reuse the stored BoVW histograms from qau_campus.vnsdb for the "
            "reference tile files under simulation/database/images_real/. This validates "
            "database/index self-retrieval integrity only; it does not prove live "
            "ROS/Gazebo/PX4 localization or fresh OpenCV feature extraction."
        ),
        f"Top-1 accuracy: {top1_hits}/{total} ({top1_pct:.2f}%)",
        f"Top-5 accuracy: {top5_hits}/{total} ({top5_pct:.2f}%)",
        "Example retrievals:",
        *example_lines,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an offline .vnsdb self-check.")
    parser.add_argument(
        "--output",
        default=str(REPO / "reports/openhouse_proof/offline_vns_check.txt"),
        help="Where to write the self-check report.",
    )
    args = parser.parse_args()

    report = build_report()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
