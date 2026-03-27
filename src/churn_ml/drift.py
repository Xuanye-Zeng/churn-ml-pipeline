"""Drift monitoring between training runs.

Compares the current run's data quality report with the previous
run's to flag distribution changes that might affect model quality.
"""

from __future__ import annotations

import json
from pathlib import Path


def find_previous_report(run_paths) -> dict | None:
    """Load the data quality report from the most recent previous run."""
    runs_dir = run_paths.output_dir / "runs"
    if not runs_dir.exists():
        return None

    current_run_name = run_paths.run_dir.name
    run_dirs = sorted(
        d for d in runs_dir.iterdir() if d.is_dir() and d.name != current_run_name
    )
    if not run_dirs:
        return None

    previous_report_path = run_dirs[-1] / "data_quality_report.json"
    if not previous_report_path.exists():
        return None

    return json.loads(previous_report_path.read_text())


def _compare_target_distribution(current: dict, previous: dict) -> dict | None:
    current_dist = current.get("target_distribution", {})
    previous_dist = previous.get("target_distribution", {})
    if not current_dist or not previous_dist:
        return None

    current_total = sum(current_dist.values())
    previous_total = sum(previous_dist.values())
    if current_total == 0 or previous_total == 0:
        return None

    # Handle both "Yes"/1 as positive-class keys
    current_pos = current_dist.get("Yes", current_dist.get(1, 0))
    previous_pos = previous_dist.get("Yes", previous_dist.get(1, 0))
    current_rate = current_pos / current_total
    previous_rate = previous_pos / previous_total
    delta = current_rate - previous_rate

    return {
        "current_positive_rate": round(current_rate, 4),
        "previous_positive_rate": round(previous_rate, 4),
        "delta": round(delta, 4),
        "flagged": abs(delta) > 0.05,
    }


def _compare_missing_rates(current: dict, previous: dict) -> list[dict]:
    current_rates = current.get("missing_rates", {})
    previous_rates = previous.get("missing_rates", {})
    flags = []

    for column in current_rates:
        curr = current_rates.get(column, 0)
        prev = previous_rates.get(column, 0)
        delta = curr - prev
        if abs(delta) > 0.01:
            flags.append(
                {
                    "column": column,
                    "current_rate": curr,
                    "previous_rate": prev,
                    "delta": round(delta, 4),
                }
            )

    return flags


def _compare_row_counts(current: dict, previous: dict) -> dict:
    curr = current.get("row_count", 0)
    prev = previous.get("row_count", 0)
    delta = curr - prev
    pct_change = round(delta / prev, 4) if prev > 0 else 0.0

    return {
        "current_rows": curr,
        "previous_rows": prev,
        "delta": delta,
        "pct_change": pct_change,
        "flagged": abs(pct_change) > 0.1,
    }


def build_drift_report(run_paths, current_report: dict) -> dict | None:
    """Compare current data quality with the previous run.

    Returns None if there is no previous run to compare against.
    """
    previous_report = find_previous_report(run_paths)
    if previous_report is None:
        return None

    target_drift = _compare_target_distribution(current_report, previous_report)
    missing_drift = _compare_missing_rates(current_report, previous_report)
    row_drift = _compare_row_counts(current_report, previous_report)

    has_drift = (
        (target_drift is not None and target_drift["flagged"])
        or len(missing_drift) > 0
        or row_drift["flagged"]
    )

    return {
        "has_drift": has_drift,
        "row_count": row_drift,
        "target_distribution": target_drift,
        "missing_rate_changes": missing_drift,
    }
