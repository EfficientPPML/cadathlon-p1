"""Hidden cases (C5/C6) are scored separately from the public C3/C4 and folded into one overall verdict."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

SUITE = Path(__file__).resolve().parents[1] / 'non_agent_pset/p1_benchmark/src/suite'
sys.path.insert(0, str(SUITE))
spec = importlib.util.spec_from_file_location('hidden_score_test', SUITE / 'score.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
spec = importlib.util.spec_from_file_location('hidden_overall_test', SUITE / 'overall.py')
overall = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overall)
overall.score = score                      # the module under test must share the patched SUITE_DIR
import evaluate
import install_suite
import rank

PUBLIC, HIDDEN = ['C3', 'C4'], ['C5', 'C6']


def test_hidden_schedule_is_declared_disjoint_from_the_public_one():
    public, hidden = score.schedule('student'), score.schedule('student', True)
    assert public['case_ids'] == PUBLIC and hidden['case_ids'] == HIDDEN
    assert not set(public['case_ids']) & set(hidden['case_ids'])
    assert public['budget_cap_s'] == 300 and hidden['budget_cap_s'] == 1800
    assert (public['baseline_label'], public['candidate_label']) == ('student-baseline', 'student-candidate')
    assert (hidden['baseline_label'], hidden['candidate_label']) == ('student-hidden-baseline', 'student-hidden-candidate')
    assert public['manifest'].name == 'baseline_manifest_student.json'
    assert hidden['manifest'].name == 'baseline_manifest_student_hidden.json'
    legacy = score.schedule('legacy')
    assert legacy['case_ids'] == ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'] and legacy['manifest'].name == 'baseline_manifest.json'
    assert (legacy['baseline_label'], legacy['candidate_label']) == ('baseline', 'candidate')
    with pytest.raises(ValueError, match='no hidden cases'):
        score.schedule('legacy', True)
    # every hidden case is a known ranked case with the student ring's device counts
    cases = {c['id']: c for c in score.CONFIG['cases']}
    assert [(cases[c]['batch'], cases[c]['devices']) for c in HIDDEN] == [(1000, 1), (1000, 4)]


def test_overlapping_or_unknown_hidden_cases_are_rejected(monkeypatch):
    config = copy.deepcopy(score.CONFIG)
    config['evaluation_profiles']['student']['hidden_case_ids'] = ['C4', 'C5']
    monkeypatch.setattr(score, 'CONFIG', config)
    with pytest.raises(ValueError, match='overlap'):
        score.schedule('student', True)
    config['evaluation_profiles']['student']['hidden_case_ids'] = ['C9']
    with pytest.raises(ValueError, match='known cases'):
        score.schedule('student', True)


@pytest.mark.xfail(strict=True, raises=ValueError, reason=(
    'suite_config.json changed on 2026-11-08 (hidden C5/C6), so the shipped 2026-11-07 student proof no longer '
    'validates. Re-run `make suite-evaluate-baseline` on the v6e-4 host, commit the new manifest and proof, then '
    'delete this marker so the test guards later config edits.'))
def test_shipped_student_calibration_is_bound_to_the_current_config():
    """The shipped manifest's proof must validate against this checkout's config, model and archives."""
    manifest = json.loads((SUITE / 'baseline_manifest_student.json').read_text())
    proof = SUITE / 'calibration' / manifest['evaluation_label'] / manifest['evaluation_id'] / 'report.json'
    report = json.loads(proof.read_text())
    score.validate_student_evaluation(report, label=manifest['evaluation_label'], mode='baseline',
                                      folders={cid: proof.parent / 'archives' / cid for cid in PUBLIC})


@pytest.mark.parametrize('key,hidden', [('budget_s', False), ('hidden_budget_s', True), ('budget_s', None)])
def test_every_schedule_must_declare_its_budget_cap(monkeypatch, key, hidden):
    """A missing budget key must not silently uncap the student schedule; null is the explicit 'no cap'."""
    config = copy.deepcopy(score.CONFIG)
    profile = 'legacy' if hidden is None else 'student'
    del config['evaluation_profiles'][profile][key]
    monkeypatch.setattr(score, 'CONFIG', config)
    with pytest.raises(ValueError, match=f'must declare {key}'):
        score.schedule(profile, bool(hidden))
    monkeypatch.undo()
    assert score.schedule('legacy')['budget_cap_s'] is None          # declared null: legacy stays uncapped
    config = copy.deepcopy(score.CONFIG)
    config['evaluation_profiles']['student']['budget_s'] = True
    monkeypatch.setattr(score, 'CONFIG', config)
    with pytest.raises(ValueError, match='positive number'):
        score.schedule('student')


@pytest.fixture
def measured(tmp_path, monkeypatch):
    """Synthetic public and hidden evaluations for a baseline and a candidate, with both calibrations."""
    suite = tmp_path / 'problem/src/suite'
    suite.mkdir(parents=True)
    monkeypatch.setattr(score, 'SUITE_DIR', suite)
    config = suite / 'suite_config.json'
    config.write_text(json.dumps(score.CONFIG))
    weights = suite.parent.parent / 'weights'
    weights.mkdir()
    (weights / 'mlp_784_50_10.pth').write_bytes(b'frozen-public-weights')
    (weights / 'MANIFEST.json').write_text(json.dumps({'files': {'mlp_784_50_10.pth': {
        'sha256': hashlib.sha256(b'frozen-public-weights').hexdigest(), 'bytes': 21}}}))
    model = score.model_identity(suite.parent.parent)
    versions = {name: 'test-version' for name in ('jax', 'jaxlib', 'libtpu', 'numpy', 'torch', 'python')}
    hashes = {'problem/src/bench_opt.py': 'e' * 64, 'problem/suite/suite_config.json': score.sha256(config)}
    # candidate compute per case: public 25% faster, hidden C5 11.1% faster, C6 25% faster -> min ratio 1.111
    compute = {'baseline': {cid: 0.5 for cid in PUBLIC + HIDDEN},
               'candidate': {'C3': 0.4, 'C4': 0.4, 'C5': 0.45, 'C6': 0.4}}
    reports = {}
    for hidden in (False, True):
        plan = score.schedule('student', hidden)
        for mode in ('baseline', 'candidate'):
            label = plan[f'{mode}_label']
            run_id = hashlib.sha256(label.encode()).hexdigest()[:32]
            report = dict(profile='student', schedule=plan['name'], gate_version=score.STUDENT_GATE_VERSION, mode=mode,
                          label=label, suite_commit=score.CONFIG['suite']['commit'], seed=score.CONFIG['determinism']['seed'],
                          runs_per_case=3, require_tpu=True, passed=True, cleanup_passed=True, within_budget=True,
                          elapsed_s=200. if not hidden else 900., budget_s=float(plan['budget_cap_s']),
                          evaluation_id=run_id, model_identity=model, cases={}, archive_sha256={}, versions=dict(versions),
                          source_sha256=dict(hashes), source_sha256_after=dict(hashes), source_files_unchanged=True)
            for cid in plan['case_ids']:
                case = next(c for c in score.CONFIG['cases'] if c['id'] == cid)
                folder = suite / 'results' / f'{cid}_{label}'
                folder.mkdir(parents=True)
                meta = dict(case, baseline_profile='student', label=label, evaluation_id=run_id,
                            suite_commit=score.CONFIG['suite']['commit'], seed=score.CONFIG['determinism']['seed'],
                            identity={'model': model, 'pixels_sha256': '1' * 64, 'labels_sha256': '2' * 64},
                            execution={'batch': case['batch'], 'devices': [f'TPU_{i}' for i in range(case['devices'])],
                                       'device_kinds': ['TPU test'] * case['devices'],
                                       'execution_rings': [{'security_bits': 128, 'log2_QP': 212.45}]},
                            verify={'status': 'PASS', 'complete': True, 'correct': case['batch'],
                                    'own_cleartext': {'status': 'ok', 'compared': case['batch'], 'top1_agreement': 1.,
                                                      'cleartext_correct': case['batch']}})
                record = {'Timing': {'Total': '20s', 'Key Generation': '1s', 'Encrypted model preprocessing': '2s',
                                     'Encrypted computation': f"{compute[mode][cid]}s"},
                          'Quality': {'Encrypted model quality': {'accuracy': 1., 'total_samples': case['batch']},
                                      'Harness model quality': {'accuracy': 1.}}}
                (folder / 'case.json').write_text(json.dumps(meta))
                for i in range(1, 4):
                    (folder / f'results-{i}.json').write_text(json.dumps(record))
                report['cases'][cid] = dict(score.summarize([record] * 3, case['batch'], meta), passed=True)
                report['archive_sha256'][cid] = score.archive_hashes(folder, 3)
            path = score.evaluation_path(label)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report))
            if mode == 'baseline':
                snapshot = score.calibration_report_path(label, run_id)
                snapshot.parent.mkdir(parents=True)
                snapshot.write_text(json.dumps(report))
                for cid in plan['case_ids']:
                    shutil.copytree(suite / 'results' / f'{cid}_{label}', snapshot.parent / 'archives' / cid)
                plan['manifest'].write_text(json.dumps(score.calibration_payload(report['cases'], 'student', report, hidden)))
            reports[(plan['name'], mode)] = report
    return suite, reports


def run_score(monkeypatch, capsys, *extra):
    monkeypatch.setattr(sys, 'argv', ['score', '--evaluation-profile', 'student', *extra])
    code = score.main()
    return code, json.loads(capsys.readouterr().out)


def test_hidden_schedule_scores_against_its_own_calibration(measured, monkeypatch, capsys):
    code, report = run_score(monkeypatch, capsys, '--hidden')
    assert code == 0 and report['passed'], report
    assert report['schedule'] == 'hidden' and report['ranked_case_ids'] == HIDDEN and report['label'] == 'student-hidden-candidate'
    assert set(report['cases']) == set(HIDDEN)
    assert report['cases']['C5']['throughput_ratio'] == pytest.approx(0.5 / 0.45)
    assert report['cases']['C6']['throughput_ratio'] == pytest.approx(1.25)
    code, public = run_score(monkeypatch, capsys)
    assert code == 0 and public['schedule'] == 'public' and set(public['cases']) == set(PUBLIC)


def test_public_evidence_cannot_be_scored_as_hidden(measured, monkeypatch, capsys):
    code, report = run_score(monkeypatch, capsys, '--hidden', '--evaluation-report', str(score.evaluation_path('student-candidate')))
    assert code == 1 and report['points_before_ranking_bonus'] == 0
    assert 'label' in report['evaluation_error'] or 'schedule' in report['evaluation_error']


def test_hidden_report_relabelled_as_public_is_rejected(measured, monkeypatch, capsys):
    suite, reports = measured
    report = dict(reports[('hidden', 'candidate')], label='student-candidate')
    score.evaluation_path('student-candidate').write_text(json.dumps(report))
    code, result = run_score(monkeypatch, capsys)
    assert code == 1 and 'schedule' in result['evaluation_error']


def test_public_calibration_cannot_serve_the_hidden_schedule(measured, monkeypatch, capsys):
    code, report = run_score(monkeypatch, capsys, '--hidden', '--baseline', str(score.schedule('student')['manifest']))
    assert code == 1 and 'other schedule' in report['baseline_error']
    assert report['improvement_points'] == 0


def test_hidden_budget_cap_is_enforced(measured, monkeypatch, capsys):
    suite, reports = measured
    report = dict(reports[('hidden', 'candidate')], budget_s=1801.)
    score.evaluation_path('student-hidden-candidate').write_text(json.dumps(report))
    code, result = run_score(monkeypatch, capsys, '--hidden')
    assert code == 1 and 'at most 1800' in result['evaluation_error']


def run_overall(monkeypatch, capsys, *extra):
    monkeypatch.setattr(sys, 'argv', ['overall', '--evaluation-profile', 'student', *extra])
    code = overall.main()
    return code, json.loads(capsys.readouterr().out)


def test_overall_applies_the_policy_over_public_and_hidden_cases(measured, monkeypatch, capsys):
    code, report = run_overall(monkeypatch, capsys)
    assert code == 0 and report['passed'], report
    assert report['ranked_case_ids'] == PUBLIC + HIDDEN and report['evaluation_profile'] == 'student+hidden'
    assert report['binding'] == {'same_sources': True, 'same_model': True, 'same_versions': True}
    assert report['correctness_points'] == 3.0 and all(c['points'] == 0.75 for c in report['cases'].values())
    assert report['improvement_min_ratio'] == pytest.approx(0.5 / 0.45)
    assert report['improvement_points'] == 2 and report['ranking_eligible'] is True
    ratios = [1.25, 1.25, 0.5 / 0.45, 1.25]
    assert report['ranking_metric'] == pytest.approx(round((ratios[0] * ratios[1] * ratios[2] * ratios[3]) ** 0.25, 4))
    assert report['points_before_ranking_bonus'] == 5.0
    assert report['policy'] == score.POLICY and len(report['baseline_manifest_sha256']) == 64


def test_overall_refuses_schedules_from_different_submissions(measured, monkeypatch, capsys):
    suite, reports = measured
    report = reports[('hidden', 'candidate')]
    report['source_sha256']['problem/src/bench_opt.py'] = 'f' * 64
    report['source_sha256_after'] = dict(report['source_sha256'])
    score.evaluation_path('student-hidden-candidate').write_text(json.dumps(report))
    code, result = run_overall(monkeypatch, capsys)
    assert code == 1 and not result['passed']
    assert any('same submission' in e for e in result['errors'])
    assert result['binding']['same_sources'] is False
    assert result['correctness_points'] == 0 and result['improvement_points'] == 0 and result['ranking_eligible'] is False


def test_overall_without_a_hidden_run_keeps_only_public_correctness(measured, monkeypatch, capsys):
    score.evaluation_path('student-hidden-candidate').unlink()
    code, result = run_overall(monkeypatch, capsys)
    assert code == 1 and not result['passed'] and result['missing_evaluations'] == ['hidden']
    assert result['evaluations'] == {'public': 'ok', 'hidden': 'missing'} and result['errors'] == []
    assert result['correctness_points'] == 1.5 and result['improvement_points'] == 0 and result['ranking_eligible'] is False
    assert result['cases']['C3']['passed'] == 1 and result['cases']['C5']['passed'] == 0
    assert any('missing' in reason for reason in result['cases']['C5']['reasons'])


def test_interrupted_hidden_run_is_not_blamed_as_a_different_submission(measured, monkeypatch, capsys, tmp_path):
    """A hidden whole-run report the evaluator wrote but never completed costs only the hidden cases.

    The artifact is produced by evaluate.run itself, failing during setup before it
    records source hashes, the model identity and software versions, which is also
    what the report holds while an evaluation is still running.
    """
    suite, _ = measured
    monkeypatch.setattr(evaluate, 'SUITE', suite)
    monkeypatch.setattr(evaluate, 'cleanup', lambda: None)
    monkeypatch.setenv('CROSS_ROOT', str(tmp_path / 'no-cross'))

    def version(name):
        raise evaluate.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(evaluate.importlib.metadata, 'version', version)
    args = SimpleNamespace(profile='student', hidden=True, baseline=False, budget_s=1800., suite_root=tmp_path, report=None)
    assert evaluate.run(args) == 1
    capsys.readouterr()                                              # drop the evaluator's own one-line verdict
    written = json.loads(score.evaluation_path('student-hidden-candidate').read_text())
    assert written['passed'] is False and 'source_sha256' not in written and written['versions'] == {}
    code, result = run_overall(monkeypatch, capsys)
    assert code == 1 and not result['passed']
    assert result['evaluations'] == {'public': 'ok', 'hidden': 'incomplete'} and result['errors'] == []
    assert result['binding'] == {'same_sources': None, 'same_model': None, 'same_versions': None}
    assert result['correctness_points'] == 1.5 and result['improvement_points'] == 0
    assert result['cases']['C3']['passed'] == 1 and result['cases']['C5']['passed'] == 0
    assert any('incomplete' in reason for reason in result['cases']['C5']['reasons'])
    # the evaluator's pre-work placeholder (what a still-running hidden run leaves on disk) is treated the same
    placeholder = {'passed': False, 'profile': 'student', 'schedule': 'hidden', 'label': 'student-hidden-candidate',
                   'versions': {}, 'cases': {}, 'error': 'evaluation has not completed'}
    public_eval = json.loads(score.evaluation_path('student-candidate').read_text())
    public_report, _ = score.build_report(score.parse_args(['--evaluation-profile', 'student']))
    hidden_report, _ = score.build_report(score.parse_args(['--evaluation-profile', 'student', '--hidden']))
    combined = overall.combine(public_report, hidden_report, public_eval, placeholder, profile='student')
    assert combined['evaluations']['hidden'] == 'incomplete' and combined['correctness_points'] == 1.5


def test_overall_reports_rank_together_but_not_with_schedule_reports(measured, monkeypatch, capsys, tmp_path):
    code, alice = run_overall(monkeypatch, capsys)
    assert code == 0
    bob = copy.deepcopy(alice)
    bob['ranking_metric'] = round(alice['ranking_metric'] - 0.05, 4)
    bob['improvement_points'] = 1
    bob['points_before_ranking_bonus'] = 4.0
    paths = {}
    for name, report in (('alice', alice), ('bob', bob)):
        paths[name] = tmp_path / name / 'overall.json'
        paths[name].parent.mkdir()
        paths[name].write_text(json.dumps(report))
    board = rank.leaderboard([rank.load_report(f'{name}={path}') for name, path in paths.items()])
    by = {t['team']: t for t in board['teams']}
    assert by['alice']['ranking_bonus'] == 3 and by['alice']['total_points'] == 8.0
    assert by['bob']['ranking_bonus'] == 2 and by['bob']['total_points'] == 6.0
    _, public_only = run_score(monkeypatch, capsys)
    with pytest.raises(ValueError, match='ranked_case_ids|evaluation_profile'):
        rank.leaderboard([alice, dict(public_only, team='carol')])


def test_evaluate_hidden_uses_its_own_label_cases_and_manifest(monkeypatch, tmp_path):
    suite = tmp_path / 'checkout/cadathlon/non_agent_pset/p1/src/suite'
    suite.mkdir(parents=True)
    monkeypatch.setattr(evaluate, 'SUITE', suite)
    monkeypatch.setattr(evaluate.score, 'SUITE_DIR', suite)       # the canonical score module evaluate uses
    monkeypatch.setattr(evaluate, 'model_identity', lambda *args: {})
    monkeypatch.setattr(evaluate, 'cleanup', lambda: None)
    monkeypatch.setattr(evaluate.importlib.metadata, 'version', lambda name: 'test')
    monkeypatch.setattr(install_suite, 'harness_is_pristine', lambda root: (False, 'changed harness'))
    args = SimpleNamespace(profile='student', hidden=True, baseline=False, budget_s=1800., suite_root=tmp_path, report=None)
    assert evaluate.run(args) == 1
    report = json.loads((suite / 'run/evaluation-student-hidden-candidate/report.json').read_text())
    assert report['schedule'] == 'hidden' and report['label'] == 'student-hidden-candidate'
    assert report['case_ids'] == HIDDEN and 'changed harness' in report['error']
    args = SimpleNamespace(profile='student', hidden=True, baseline=False, budget_s=1801., suite_root=tmp_path, report=None)
    assert evaluate.run(args) == 1
    report = json.loads((suite / 'run/evaluation-student-hidden-candidate/report.json').read_text())
    assert 'at most 1800' in report['error']


def test_hidden_calibration_is_published_next_to_the_evaluator(monkeypatch, tmp_path):
    """A passing hidden baseline run publishes baseline_manifest_student_hidden.json under evaluate.SUITE."""
    suite = tmp_path / 'checkout/cadathlon/non_agent_pset/p1/src/suite'
    scorer_root = tmp_path / 'scorer/src/suite'                       # deliberately a different root
    for folder in (suite, scorer_root):
        folder.mkdir(parents=True)
    monkeypatch.setattr(evaluate, 'SUITE', suite)
    monkeypatch.setattr(evaluate.score, 'SUITE_DIR', scorer_root)
    monkeypatch.setattr(evaluate, 'model_identity', lambda *args: {})
    monkeypatch.setattr(evaluate, 'archive_hashes', lambda *args: {})
    monkeypatch.setattr(evaluate, 'cleanup', lambda: None)
    monkeypatch.setattr(evaluate, 'run_command', lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate.importlib.metadata, 'version', lambda name: 'test')
    monkeypatch.setattr(install_suite, 'harness_is_pristine', lambda root: (True, ''))
    monkeypatch.setattr(install_suite, 'install', lambda root: None)
    monkeypatch.setattr(evaluate.score, 'read_case', lambda *args: ([], {}, tmp_path))
    monkeypatch.setattr(evaluate.score, 'summarize', lambda *args: {'baseline_profile': 'student'})
    monkeypatch.setattr(evaluate.score, 'gate', lambda *args: (1, []))
    validated = []
    monkeypatch.setattr(evaluate.score, 'validate_student_evaluation',
                        lambda report, **kwargs: validated.append((kwargs.get('hidden'), kwargs.get('mode'))) or report)
    seen = []
    monkeypatch.setattr(evaluate.score, 'calibration_payload',
                        lambda cases, profile, report, hidden: seen.append((sorted(cases), hidden)) or {'hidden_calibration': True})
    monkeypatch.setenv('CROSS_ROOT', str(tmp_path / 'no-cross'))
    args = SimpleNamespace(profile='student', hidden=True, baseline=True, budget_s=1800., suite_root=tmp_path, report=None)
    assert evaluate.run(args) == 0
    assert seen == [(HIDDEN, True)]
    # the run's evidence and the published proof are both validated against the HIDDEN schedule
    assert validated == [(True, None), (True, 'baseline')]
    assert json.loads((suite / 'baseline_manifest_student_hidden.json').read_text()) == {'hidden_calibration': True}
    assert not (scorer_root / 'baseline_manifest_student_hidden.json').exists()
    assert not (suite / 'baseline_manifest_student.json').exists()
    report = json.loads((suite / 'run/evaluation-student-hidden-baseline/report.json').read_text())
    assert report['passed'] is True and report['schedule'] == 'hidden' and report['label'] == 'student-hidden-baseline'
    assert (scorer_root / 'calibration/student-hidden-baseline' / report['evaluation_id'] / 'report.json').is_file()


@pytest.fixture
def evaluate_cli(monkeypatch, tmp_path):
    """evaluate.main() in-process: run() only records its arguments, nothing touches the real checkout."""
    import contextlib
    monkeypatch.setattr(evaluate, 'SUITE', tmp_path)
    monkeypatch.setattr(evaluate, 'evaluation_lock', contextlib.nullcontext)
    monkeypatch.setattr(evaluate, 'cancellation_handlers', contextlib.nullcontext)
    calls = []
    monkeypatch.setattr(evaluate, 'run', lambda args: calls.append(args) or 0)

    def main(*argv):
        monkeypatch.setattr(sys, 'argv', ['evaluate.py', *argv])
        return evaluate.main()
    return main, calls


@pytest.mark.parametrize('argv,budget', [
    ((), 300.), (('--baseline',), 300.), (('--hidden',), 1800.), (('--hidden', '--baseline'), 1800.),
    (('--profile', 'legacy'), 300.), (('--hidden', '--budget-s', '900'), 900.),
    (('--profile', 'legacy', '--budget-s', '7200'), 7200.)])
def test_evaluate_budget_defaults_to_the_schedule_cap(evaluate_cli, argv, budget):
    """The Make targets pass no --budget-s, so these defaults are what the organizer runs."""
    main, calls = evaluate_cli
    assert main(*argv) == 0
    assert calls[-1].budget_s == budget and calls[-1].hidden is ('--hidden' in argv)


@pytest.mark.parametrize('argv,message', [
    (('--hidden', '--budget-s', '1801'), 'at most 1800'), (('--budget-s', '301'), 'at most 300'),
    (('--profile', 'legacy', '--hidden'), 'no hidden cases')])
def test_evaluate_cli_rejects_relaxed_budgets_before_any_work(evaluate_cli, capsys, argv, message):
    main, calls = evaluate_cli
    with pytest.raises(SystemExit) as exc:
        main(*argv)
    assert exc.value.code == 2 and message in capsys.readouterr().err and calls == []


def make_dry_run(*targets, directory=SUITE.parent):
    """One string per recipe command, with backslash continuations joined."""
    output = subprocess.check_output(['make', '-n', '--no-print-directory', '-C', str(directory), *targets], text=True)
    return [' '.join(line.split()) for line in output.replace('\\\n', ' ').splitlines() if 'suite/' in line]


def test_make_targets_drive_the_hidden_schedule_and_the_overall_script():
    calibrate, evaluate_hidden, score_hidden, combine = make_dry_run(
        'suite-evaluate-baseline-hidden', 'suite-evaluate-hidden', 'suite-score-hidden', 'suite-overall')
    # the budget comes from suite_config.json unless the organizer lowers it
    assert calibrate.endswith('evaluate.py --profile student --hidden --baseline')
    assert evaluate_hidden.endswith('evaluate.py --profile student --hidden')
    assert '--hidden --label student-hidden-candidate --baseline' in score_hidden
    assert score_hidden.split('--baseline ')[1].split()[0].endswith('/baseline_manifest_student_hidden.json')
    words = combine.split()
    assert words[words.index('--public-label') + 1] == 'student-candidate'
    assert words[words.index('--hidden-label') + 1] == 'student-hidden-candidate'
    assert words[words.index('--public-baseline') + 1].endswith('/baseline_manifest_student.json')
    assert words[words.index('--hidden-baseline') + 1].endswith('/baseline_manifest_student_hidden.json')
    lowered, = make_dry_run('suite-evaluate-hidden', 'HIDDEN_BUDGET_S=900')
    assert lowered.endswith('--hidden --budget-s 900')
    # the public targets also leave the budget to suite_config.json; legacy still takes an explicit one
    public_run, public_calibration = make_dry_run('suite-evaluate', 'suite-evaluate-baseline')
    assert public_run.endswith('evaluate.py --profile student')
    assert public_calibration.endswith('evaluate.py --profile student --baseline')
    legacy, = make_dry_run('suite-evaluate', 'P1_BASELINE_PROFILE=legacy', 'EVALUATION_BUDGET_S=7200')
    assert legacy.endswith('evaluate.py --profile legacy --budget-s 7200')
    overridden, = make_dry_run('suite-overall', 'BASELINE_MANIFEST=/judge/public.json', 'HIDDEN_BASELINE_MANIFEST=/judge/hidden.json')
    assert '--public-baseline /judge/public.json' in overridden and '--hidden-baseline /judge/hidden.json' in overridden
    root = subprocess.check_output(['make', '-n', '-C', str(SUITE.parents[4]), 'suite-overall'], text=True)
    assert 'suite/overall.py' in root


# Every file a hidden calibration and a hidden candidate run leave in the checkout.
HIDDEN_ARTIFACTS = (
    'baseline_manifest_student_hidden.json',
    'calibration/student-hidden-baseline/def/report.json',
    'calibration/student-hidden-baseline/def/archives/C5/case.json',
    'calibration/student-hidden-baseline/def/archives/C6/results-1.json',
    'results/C5_student-hidden-baseline/case.json',
    'results/C5_student-hidden-baseline/results-1.json',
    'results/C6_student-hidden-baseline/results-3.json',
    'results/C5_student-hidden-candidate/results-2.json',
    'results/C6_student-hidden-candidate/case.json',
    'run/evaluation-student-hidden-baseline/report.json',
)
PUBLIC_ARTIFACTS = (
    'baseline_manifest_student.json',
    'calibration/student-baseline/abc/report.json',
    'results/C3_student-baseline/results-1.json',
    'results/C5_baseline/results-1.json',                              # the legacy ring's published C5
)


def test_release_excludes_every_hidden_artifact_but_ships_the_overall_script(tmp_path):
    """package.py release drops the hidden manifest, proof tree and archives, and keeps overall.py."""
    import tarfile
    root = tmp_path / 'cadathlon'
    problem = root / 'non_agent_pset/p1_benchmark'
    common = SUITE.parents[3] / 'common'
    (problem / 'src/suite').mkdir(parents=True)
    (root / 'common').mkdir()
    for name in ('package.py', 'support.py', 'Makefile.problem'):
        shutil.copy2(common / name, root / 'common' / name)
    shutil.copy2(SUITE.parent.parent / 'EXCLUDE', problem / 'EXCLUDE')
    shutil.copy2(SUITE / 'overall.py', problem / 'src/suite/overall.py')
    for rel in HIDDEN_ARTIFACTS + PUBLIC_ARTIFACTS:
        path = problem / 'src/suite' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    (root / 'validation/p1_student_v6e8').mkdir(parents=True)
    for name in ('p1_student_suite.json', 'p1_student_v6e8/report.json', 'p1_legacy_suite.json'):
        (root / 'validation' / name).write_text('{}')
    subprocess.run([sys.executable, str(root / 'common/package.py'), 'release', str(root)], check=True)
    with tarfile.open(root / 'contestant-release.tar.gz') as archive:
        names = set(archive.getnames())
    prefix = 'non_agent_pset/p1_benchmark/src/suite/'
    assert prefix + 'overall.py' in names
    assert {prefix + rel for rel in PUBLIC_ARTIFACTS} <= names
    assert not any('hidden' in name for name in names), sorted(n for n in names if 'hidden' in n)


def test_git_ignores_every_hidden_artifact(tmp_path):
    """The hidden baseline must not be committed by accident either."""
    repo = SUITE.parents[4]
    paths = ['cadathlon/non_agent_pset/p1_benchmark/src/suite/' + rel for rel in HIDDEN_ARTIFACTS]
    ignored = subprocess.run(['git', '-C', str(repo), 'check-ignore', '--no-index', *paths],
                             capture_output=True, text=True).stdout.split()
    assert sorted(ignored) == sorted(paths)
    public = ['cadathlon/non_agent_pset/p1_benchmark/src/suite/' + rel for rel in PUBLIC_ARTIFACTS]
    assert subprocess.run(['git', '-C', str(repo), 'check-ignore', '--no-index', *public],
                          capture_output=True, text=True).stdout.split() == []
