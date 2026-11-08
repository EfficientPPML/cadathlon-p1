"""Assign the P1 ranking bonus across teams from their saved score.py reports.

Each team's `suite-score` output already carries its correctness and
improvement points (policy p1-10-point-v1, handout section 4). This offline
step adds the only cross-team part: the three best *eligible* submissions,
ordered by their ranking metric (the geometric mean of the per-case throughput
ratios), receive 3, 2 and 1 bonus points. A submission is eligible only if it
earned at least one improvement point, i.e. it is at least 5% faster than the
baseline on every ranked case. Metrics are compared after rounding to four
decimals; teams with equal rounded metrics share the higher bonus and the
following rank is skipped.

    python3 rank.py --report alice=/abs/teams/alice/score.json --report bob=/abs/teams/bob/score.json

Use absolute paths (the Make target runs from `src/`). A report path without a
`team=` prefix is named after its parent directory. Reports must come from one
schedule, one policy and one calibration manifest; the tool refuses to mix them,
lists tied teams alphabetically and never redistributes an unclaimed bonus.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score

SHARED = ("policy", "evaluation_profile", "suite_commit", "seed", "ranked_case_ids", "baseline_manifest_sha256")


def load_report(spec):
    team, sep, path = spec.partition("=")
    # "team=path": the prefix is a team name only when it is not itself a path.
    if not sep or "/" in team or Path(spec).is_file():
        team, path = "", spec
    path = Path(path)
    text = path.read_text()
    # `make suite-score` output may carry make's own echo lines before the JSON.
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"{path}: no JSON object found")
    report = json.loads(text[start:end + 1])
    if not isinstance(report, dict):
        raise ValueError(f"{path}: report must be a JSON object")
    try:
        json.dumps(report, allow_nan=False)
    except ValueError as error:
        raise ValueError(f"{path}: report contains NaN or infinite values") from error
    report["team"] = team or path.resolve().parent.name or path.stem
    return report


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def leaderboard(reports):
    if not reports:
        raise ValueError("at least one report is required")
    names = [r.get("team") for r in reports]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate team names: {sorted(n for n in names if names.count(n) > 1)}")
    for field in SHARED:
        values = {json.dumps(r.get(field), sort_keys=True) for r in reports}
        if len(values) != 1:
            raise ValueError(f"reports disagree on {field}; rank one schedule and one policy at a time")
    if reports[0].get("policy") != score.POLICY:
        raise ValueError(f"reports use policy {reports[0].get('policy')!r}, this tool implements {score.POLICY!r}")
    for r in reports:
        for field in ("correctness_points", "improvement_points", "points_before_ranking_bonus"):
            if not _finite(r.get(field)):
                raise ValueError(f"{r['team']}: {field} missing or invalid")
    for r in reports:
        # The flag and the points it summarises must agree; a doctored report fails here.
        if (r.get("ranking_eligible") is True) != (r["improvement_points"] >= 1):
            raise ValueError(f"{r['team']}: ranking_eligible disagrees with improvement_points")
    eligible = [r for r in reports
                if r.get("ranking_eligible") is True and r["improvement_points"] >= 1
                and _finite(r.get("ranking_metric")) and r["ranking_metric"] > 0]
    rounded = {r["team"]: round(r["ranking_metric"], score.METRIC_DECIMALS) for r in eligible}
    entries = []
    for r in reports:
        rank = bonus = None
        if r["team"] in rounded:
            mine = rounded[r["team"]]
            rank = 1 + sum(1 for other in rounded.values() if other > mine)
            bonus = score.RANKING_BONUS[rank - 1] if rank <= len(score.RANKING_BONUS) else 0
        entries.append({
            "team": r["team"],
            "passed": bool(r.get("passed")),
            "correctness_points": r["correctness_points"],
            "improvement_points": r["improvement_points"],
            "ranking_metric": r.get("ranking_metric"),
            "ranking_eligible": r["team"] in rounded,
            "rank": rank,
            "ranking_bonus": bonus or 0,
            "total_points": round(min(score.MAX_POINTS, r["points_before_ranking_bonus"] + (bonus or 0)), 3),
        })
    entries.sort(key=lambda e: (-e["total_points"], -(e["ranking_metric"] or 0.0), e["team"]))
    return {
        "policy": score.POLICY,
        "max_points": score.MAX_POINTS,
        "evaluation_profile": reports[0].get("evaluation_profile"),
        "suite_commit": reports[0].get("suite_commit"),
        "ranked_case_ids": reports[0].get("ranked_case_ids"),
        "ranking_bonus": list(score.RANKING_BONUS),
        "teams": entries,
        "note": "offline arithmetic over saved suite-score reports; the judge assigns the bonus after its own reruns",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="append", default=[], metavar="TEAM=PATH",
                        help="a saved suite-score JSON report; repeat once per team")
    args = parser.parse_args()
    if not args.report:
        parser.error("pass at least one --report TEAM=PATH")
    try:
        board = leaderboard([load_report(spec) for spec in args.report])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
        return 1
    print(json.dumps(board, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
