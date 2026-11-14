"""Fold a submission's public and hidden score reports into one overall verdict.

The public schedule (student profile: C3/C4) is what contestants run and score
themselves. The hidden schedule (student profile: C5/C6, 1000 images on one and
four chips) is run by the organizer after submission with `evaluate.py --hidden`
against its own calibration. Both whole-run evaluations and both calibration
proofs must exist in THIS checkout: the judge keeps one checkout holding both
calibrations and, one submission at a time, reruns `evaluate.py` (public) and
`evaluate.py --hidden` before running this script. It scores both schedules
with score.py's own arithmetic and re-applies policy p1-10-point-v1 over the
UNION of their cases:

  * the 3 correctness points are shared equally by every ranked case (0.75 per
    case for C3/C4 + C5/C6), so a submission that only passes the public cases
    keeps 1.5 and, lacking a passing hidden case, earns no improvement points;
  * improvement points come from the SMALLEST throughput ratio over all cases;
  * the ranking metric is the geometric mean over all cases.

It refuses to combine two complete evaluations that were not produced by the
same submission sources (every problem/CROSS source file, including
bench_opt.py), the same frozen model and the same software versions; that
fails every case. A missing or incomplete whole-run report (interrupted, still
running, or failed during setup) fails only its own schedule's cases. The
output carries the fields rank.py needs, so overall reports of several teams
can be ranked together:

    python3 overall.py --evaluation-profile student > /abs/teams/<team>/overall.json
    python3 rank.py --report <team>=/abs/teams/<team>/overall.json ...

Offline arithmetic over harness-written measurement files; not an official
score without the judge's rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score

# Whole-run report fields that identify the submission an evaluation ran.
# evaluate.py records them before any case runs, so a report without them is a
# placeholder or a run that failed during setup, never evidence of a mismatch.
BINDING = (('same_sources', 'source_sha256'), ('same_model', 'model_identity'), ('same_versions', 'versions'))


def _read_evaluation(label):
    """The raw whole-run report of `label`, or None and why it is unavailable."""
    path = score.evaluation_path(label)
    try:
        return score._read_object(path), None
    except FileNotFoundError:
        return None, f'whole-run evaluation report missing ({path})'
    except (OSError, ValueError) as error:
        return None, f'whole-run evaluation report unreadable ({error})'


def _evaluation_status(evaluation, note=None):
    """('ok' | 'missing' | 'incomplete', reason) for one schedule's whole-run report."""
    if evaluation is None:
        return 'missing', note or 'whole-run evaluation report missing'
    absent = [field for _, field in BINDING if not evaluation.get(field)]
    if absent:
        return 'incomplete', ('whole-run evaluation report is incomplete (no ' + ', '.join(absent)
                              + '): the run was interrupted, is still in progress, or failed during setup')
    return 'ok', None


def combine(public, hidden, public_evaluation, hidden_evaluation, *, profile, notes=None):
    """Overall verdict from two score reports and the whole-run reports behind them.

    `notes` maps 'public'/'hidden' to why a whole-run report is None (missing or
    unreadable); it only improves the reason text.
    """
    notes = notes or {}
    errors = []
    for name, report, want in (('public', public, 'public'), ('hidden', hidden, 'hidden')):
        if report.get('evaluation_profile') != profile or report.get('schedule') != want:
            errors.append(f'{name} report is not a {profile} {want}-schedule score report')
        if report.get('policy') != score.POLICY:
            errors.append(f'{name} report uses policy {report.get("policy")!r}, expected {score.POLICY!r}')
    shared = ('suite_commit', 'seed', 'runs_per_case', 'statistic')
    for field in shared:
        if public.get(field) != hidden.get(field):
            errors.append(f'public and hidden reports disagree on {field}')
    public_ids, hidden_ids = list(public.get('ranked_case_ids') or []), list(hidden.get('ranked_case_ids') or [])
    if set(public_ids) & set(hidden_ids) or not public_ids or not hidden_ids:
        errors.append('public and hidden schedules must be disjoint and nonempty')

    # Only two complete whole-run reports can prove or disprove that both
    # schedules ran the same submission. A missing or incomplete report fails
    # just its own schedule's cases (score.py has already refused them); a
    # provable mismatch fails everything, because then no case can be
    # attributed to one submission.
    status = {'public': _evaluation_status(public_evaluation, notes.get('public')),
              'hidden': _evaluation_status(hidden_evaluation, notes.get('hidden'))}
    unavailable = [name for name, (state, _) in status.items() if state != 'ok']
    binding = {key: None for key, _ in BINDING}
    if not unavailable:
        binding = {key: public_evaluation[field] == hidden_evaluation[field] for key, field in BINDING}
        if not all(binding.values()):
            errors.append('public and hidden evaluations were not produced by the same submission '
                          '(sources, frozen model and software versions must match)')

    ranked = public_ids + hidden_ids
    cases = {}
    for name, report in (('public', public), ('hidden', hidden)):
        for cid, verdict in (report.get('cases') or {}).items():
            entry = dict(verdict)
            entry['schedule'] = name
            reason = ('overall: ' + '; '.join(errors) if errors
                      else f'overall: {name} {status[name][1]}' if name in unavailable else None)
            if reason:
                entry['passed'] = 0
                entry['throughput_ratio'] = None
                entry['reasons'] = list(entry.get('reasons') or []) + [reason]
            entry['points'] = score.case_points(entry.get('passed'), len(ranked))
            cases[cid] = entry
    for cid in ranked:
        if cid not in cases:
            cases[cid] = {'passed': 0, 'reasons': ['case was not evaluated'], 'throughput_ratio': None, 'points': 0.0,
                          'schedule': 'public' if cid in public_ids else 'hidden'}
    points = score.policy_points(cases, ranked)
    baseline_errors = [f'{name}: {r["baseline_error"]}' for name, r in (('public', public), ('hidden', hidden)) if r.get('baseline_error')]
    evaluation_errors = [f'{name}: {r["evaluation_error"]}' for name, r in (('public', public), ('hidden', hidden)) if r.get('evaluation_error')]
    digests = [public.get('baseline_manifest_sha256') or 'none', hidden.get('baseline_manifest_sha256') or 'none']
    passed = (not errors and not unavailable and public.get('passed') is True and hidden.get('passed') is True
              and all(v.get('passed') for v in cases.values()) and not baseline_errors and not evaluation_errors)
    return {
        'passed': bool(passed),
        'evaluations': {name: state for name, (state, _) in status.items()},
        'missing_evaluations': unavailable,
        'suite_commit': public.get('suite_commit'),
        'seed': public.get('seed'),
        'runs_per_case': public.get('runs_per_case'),
        'statistic': public.get('statistic'),
        'evaluation_profile': f'{profile}+hidden',
        'schedule': 'overall',
        'public_case_ids': public_ids,
        'hidden_case_ids': hidden_ids,
        'ranked_case_ids': ranked,
        'public_label': public.get('label'),
        'hidden_label': hidden.get('label'),
        'binding': binding,
        'cases': cases,
        'errors': errors,
        'baseline_error': '; '.join(baseline_errors) or None,
        'evaluation_error': '; '.join(evaluation_errors) or None,
        'baseline_manifest_sha256_public': public.get('baseline_manifest_sha256'),
        'baseline_manifest_sha256_hidden': hidden.get('baseline_manifest_sha256'),
        # One digest over both calibrations so rank.py can refuse to mix teams
        # scored against different public/hidden manifests.
        'baseline_manifest_sha256': hashlib.sha256(':'.join(digests).encode()).hexdigest(),
        **points,
        'note': 'overall verdict over the public and hidden schedules; organizer arithmetic, '
                'not an official score without a judge rerun',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite-root', type=Path, default=None)
    parser.add_argument('--evaluation-profile', choices=sorted(score.CONFIG['evaluation_profiles']), default='student')
    parser.add_argument('--public-label', default='', help="public archive label (default <profile>-candidate)")
    parser.add_argument('--hidden-label', default='', help="hidden archive label (default <profile>-hidden-candidate)")
    parser.add_argument('--public-baseline', type=Path, default=None, help="public calibration manifest (default the profile's)")
    parser.add_argument('--hidden-baseline', type=Path, default=None, help="hidden calibration manifest (default the profile's)")
    args = parser.parse_args(argv)
    profile = args.evaluation_profile
    try:
        public_plan, hidden_plan = score.schedule(profile), score.schedule(profile, True)
    except ValueError as error:
        parser.error(str(error))
    public_label = args.public_label or public_plan['candidate_label']
    hidden_label = args.hidden_label or hidden_plan['candidate_label']

    def scored(label, hidden, baseline):
        argv = ['--evaluation-profile', profile, '--label', label]
        if args.suite_root is not None:
            argv += ['--suite-root', str(args.suite_root)]
        if hidden:
            argv.append('--hidden')
        if baseline is not None:
            argv += ['--baseline', str(baseline)]
        report, _ = score.build_report(score.parse_args(argv))
        return report

    public = scored(public_label, False, args.public_baseline)
    hidden = scored(hidden_label, True, args.hidden_baseline)
    public_evaluation, public_note = _read_evaluation(public_label)
    hidden_evaluation, hidden_note = _read_evaluation(hidden_label)
    overall = combine(public, hidden, public_evaluation, hidden_evaluation, profile=profile,
                      notes={'public': public_note, 'hidden': hidden_note})
    print(json.dumps(overall, indent=2, allow_nan=False))
    return 0 if overall['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
