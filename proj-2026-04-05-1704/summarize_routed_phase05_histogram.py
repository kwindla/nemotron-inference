#!/usr/bin/env python3
import json
import sys
from collections import Counter
from pathlib import Path


def top_share(values, top_n):
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    return 100.0 * sum(sorted(values, reverse=True)[:top_n]) / total


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_routed_phase05_histogram.py <jsonl>", file=sys.stderr)
        return 1

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"missing histogram file: {path}", file=sys.stderr)
        return 1

    records = []
    with path.open("r", encoding="utf-8") as handle:
      for line in handle:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    if not records:
        print(f"empty histogram file: {path}", file=sys.stderr)
        return 1

    overall_selection = Counter()
    overall_ctas = Counter()

    print(f"phase05_histogram={path}")
    print(f"layer_count={len(records)}")
    for record in records:
        experts = record.get("experts", [])
        selection_counts = [entry["selection_count"] for entry in experts]
        cta_counts = [entry["cta_count"] for entry in experts]
        for entry in experts:
            expert_id = int(entry["expert_id"])
            overall_selection[expert_id] += int(entry["selection_count"])
            overall_ctas[expert_id] += int(entry["cta_count"])

        print(
            "layer"
            f" index={record['layer_index']}"
            f" token_count={record['token_count']}"
            f" selection_count={record['selection_count']}"
            f" active_expert_count={record['active_expert_count']}"
            f" cta_count={record['cta_count']}"
            f" total_padded_rows={record['total_padded_rows']}"
            f" top5_share={top_share(selection_counts, 5):.2f}%"
            f" top10_share={top_share(selection_counts, 10):.2f}%"
            f" max_tokens_per_expert={(max(selection_counts) if selection_counts else 0)}"
            f" max_ctas_per_expert={(max(cta_counts) if cta_counts else 0)}"
        )
        top_entries = sorted(
            experts,
            key=lambda entry: (-entry["selection_count"], -entry["cta_count"], entry["expert_id"]),
        )[:8]
        for entry in top_entries:
            print(
                "  expert"
                f" id={entry['expert_id']}"
                f" selection_count={entry['selection_count']}"
                f" cta_count={entry['cta_count']}"
                f" padded_rows=[{entry['padded_row_begin']},{entry['padded_row_end']})"
            )

    overall_selection_values = list(overall_selection.values())
    overall_cta_values = list(overall_ctas.values())
    print(
        "overall"
        f" active_experts={len(overall_selection_values)}"
        f" top5_share={top_share(overall_selection_values, 5):.2f}%"
        f" top10_share={top_share(overall_selection_values, 10):.2f}%"
        f" max_tokens_per_expert={(max(overall_selection_values) if overall_selection_values else 0)}"
        f" max_ctas_per_expert={(max(overall_cta_values) if overall_cta_values else 0)}"
    )
    top_overall = sorted(
        overall_selection.items(),
        key=lambda item: (-item[1], -overall_ctas[item[0]], item[0]),
    )[:12]
    for expert_id, selection_count in top_overall:
        print(
            "  overall_expert"
            f" id={expert_id}"
            f" selection_count={selection_count}"
            f" cta_count={overall_ctas[expert_id]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
