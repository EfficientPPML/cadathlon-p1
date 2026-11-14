"""The 10-point P1 policy: correctness per case, improvement ladder, ranking bonus."""
import json
import math
from pathlib import Path
import sys

import pytest

SUITE = Path(__file__).resolve().parents[1] / 'pset/p1_benchmark/src/suite'
sys.path.insert(0, str(SUITE))
import rank
import score

RANKED = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']


def verdict(ratios, failed=()):
    return {cid: ({'passed': 0, 'throughput_ratio': None} if cid in failed else {'passed': 1, 'throughput_ratio': r})
            for cid, r in zip(RANKED, ratios)}


@pytest.mark.parametrize('min_ratio,points', [
    (0.5, 0), (1.0, 0), (1.0499, 0), (1.05, 1), (1.0999, 1), (1.10, 2), (1.1999, 2),
    (1.20, 3), (1.3999, 3), (1.40, 4), (9.0, 4)])
def test_improvement_ladder_uses_the_smallest_ratio(min_ratio, points):
    result = score.policy_points(verdict([9.5] * 5 + [min_ratio]), RANKED)
    assert result['improvement_points'] == points
    assert result['correctness_points'] == 3.0
    assert result['points_before_ranking_bonus'] == 3.0 + points
    assert result['ranking_eligible'] is (points >= 1)
    assert result['improvement_min_ratio'] == round(min_ratio, 4)


def test_threshold_is_inclusive_despite_float_arithmetic():
    assert score.improvement_tier(105.0 / 100.0) == 1
    assert score.improvement_tier(1.05 - 1e-12) == 1
    assert score.improvement_tier(1.05 - 1e-6) == 0
    assert score.improvement_tier(None) == 0 and score.improvement_tier(float('nan')) == 0


def test_one_failed_case_forfeits_improvement_and_ranking():
    result = score.policy_points(verdict([3.0] * 6, failed={'C2'}), RANKED)
    assert result['correctness_points'] == 2.5
    assert result['improvement_points'] == 0 and result['ranking_eligible'] is False
    assert result['ranking_metric'] is None and result['points_before_ranking_bonus'] == 2.5


def test_correctness_is_shared_equally_by_the_ranked_cases():
    assert score.policy_points(verdict([1.0] * 6), RANKED)['correctness_points'] == 3.0
    assert score.policy_points(verdict([1.0] * 6, failed={'C1', 'C3', 'C5'}), RANKED)['correctness_points'] == 1.5
    two = {'C3': {'passed': 1, 'throughput_ratio': 1.0}, 'C4': {'passed': 1, 'throughput_ratio': 1.0}}
    assert score.policy_points(two, ['C3', 'C4'])['correctness_points'] == 3.0
    assert score.policy_points({**two, 'C4': {'passed': 0}}, ['C3', 'C4'])['correctness_points'] == 1.5


def test_ranking_metric_is_the_geometric_mean_of_all_ratios():
    ratios = [1.1, 1.1, 1.1, 1.1, 1.1, 4.4]
    result = score.policy_points(verdict(ratios), RANKED)
    assert result['ranking_metric'] == round(math.exp(sum(math.log(r) for r in ratios) / 6), 4)
    assert result['improvement_points'] == 2


def team(name, before, metric, eligible=True, **overrides):
    report = {'team': name, 'policy': score.POLICY, 'evaluation_profile': 'legacy',
              'suite_commit': score.CONFIG['suite']['commit'], 'seed': 3, 'ranked_case_ids': RANKED,
              'baseline_manifest_sha256': 'c' * 64, 'passed': True, 'correctness_points': 3.0, 'improvement_points': before - 3.0,
              'points_before_ranking_bonus': before, 'ranking_metric': metric, 'ranking_eligible': eligible}
    report.update(overrides)
    return report


def test_ranking_bonus_goes_to_the_three_best_eligible_teams():
    board = rank.leaderboard([team('a', 7.0, 1.8), team('b', 6.0, 1.5), team('c', 4.0, 1.2), team('d', 4.0, 1.1),
                              team('fast-but-regressed', 3.0, 2.5, eligible=False)])
    by = {t['team']: t for t in board['teams']}
    assert [by[n]['ranking_bonus'] for n in ('a', 'b', 'c', 'd', 'fast-but-regressed')] == [3, 2, 1, 0, 0]
    assert [by[n]['total_points'] for n in ('a', 'b', 'c', 'd', 'fast-but-regressed')] == [10, 8, 5, 4, 3]
    assert [t['team'] for t in board['teams']] == ['a', 'b', 'c', 'd', 'fast-but-regressed']
    assert by['fast-but-regressed']['rank'] is None


def test_tied_metrics_share_the_bonus_and_skip_the_next_rank():
    board = rank.leaderboard([team('a', 5.0, 1.23451), team('b', 5.0, 1.23449), team('c', 4.0, 1.2), team('d', 4.0, 1.19)])
    by = {t['team']: t for t in board['teams']}
    assert by['a']['rank'] == by['b']['rank'] == 1 and by['a']['ranking_bonus'] == by['b']['ranking_bonus'] == 3
    assert by['c']['rank'] == 3 and by['c']['ranking_bonus'] == 1
    assert by['d']['rank'] == 4 and by['d']['ranking_bonus'] == 0


def test_total_never_exceeds_ten():
    assert rank.leaderboard([team('a', 7.0, 3.0)])['teams'][0]['total_points'] == 10
    # an impossible 8.0 before the bonus still clamps at the policy maximum
    assert rank.leaderboard([team('a', 8.0, 3.0)])['teams'][0]['total_points'] == 10


def test_no_eligible_team_means_no_bonus_for_anyone():
    board = rank.leaderboard([team('a', 3.0, 1.0, eligible=False), team('b', 2.5, None, eligible=False)])
    assert all(t['rank'] is None and t['ranking_bonus'] == 0 for t in board['teams'])
    assert [t['total_points'] for t in board['teams']] == [3.0, 2.5]


def test_two_case_schedule_uses_the_smaller_ratio_and_their_geometric_mean():
    two = {'C3': {'passed': 1, 'throughput_ratio': 1.21}, 'C4': {'passed': 1, 'throughput_ratio': 1.44}}
    result = score.policy_points(two, ['C3', 'C4'])
    assert result['correctness_points'] == 3.0 and result['improvement_points'] == 3
    assert result['improvement_min_ratio'] == 1.21
    assert result['ranking_metric'] == round(math.sqrt(1.21 * 1.44), 4)


def test_inconsistent_eligibility_flag_is_rejected():
    with pytest.raises(ValueError, match='ranking_eligible'):
        rank.leaderboard([team('a', 3.0, 1.5, eligible=True)])          # flag set, zero improvement points
    with pytest.raises(ValueError, match='ranking_eligible'):
        rank.leaderboard([team('a', 5.0, 1.5, eligible=False)])         # two improvement points, flag cleared


def test_all_reports_from_an_unknown_policy_are_rejected():
    with pytest.raises(ValueError, match='this tool implements'):
        rank.leaderboard([team('a', 4.0, 1.1, policy='other'), team('b', 4.0, 1.1, policy='other')])


def test_rank_error_paths(tmp_path, monkeypatch, capsys):
    empty = tmp_path / 'empty.json'
    empty.write_text('make: nothing here\n')
    with pytest.raises(ValueError, match='no JSON object'):
        rank.load_report(str(empty))
    nan = tmp_path / 'nan' / 'score.json'
    nan.parent.mkdir()
    nan.write_text(json.dumps(team('nan', 4.0, float('nan'))))
    with pytest.raises(ValueError, match='NaN'):
        rank.load_report(str(nan))
    with pytest.raises(ValueError, match='correctness_points'):
        rank.leaderboard([team('a', 4.0, 1.1, correctness_points=None)])
    monkeypatch.setattr(sys, 'argv', ['rank', '--report', str(empty)])
    assert rank.main() == 1
    assert 'error' in json.loads(capsys.readouterr().out)


def test_report_path_containing_an_equals_sign_is_named_after_its_folder(tmp_path):
    path = tmp_path / 'run=2' / 'score.json'
    path.parent.mkdir()
    path.write_text(json.dumps(team('ignored', 4.0, 1.1)))
    assert rank.load_report(str(path))['team'] == 'run=2'


@pytest.mark.parametrize('field,value,match', [
    ('evaluation_profile', 'student', 'evaluation_profile'), ('policy', 'other', 'policy'),
    ('ranked_case_ids', ['C3', 'C4'], 'ranked_case_ids'), ('baseline_manifest_sha256', 'd' * 64, 'baseline_manifest_sha256')])
def test_reports_from_different_schedules_or_policies_cannot_be_ranked_together(field, value, match):
    with pytest.raises(ValueError, match=match):
        rank.leaderboard([team('a', 4.0, 1.1), team('b', 4.0, 1.1, **{field: value})])


def test_duplicate_team_names_are_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        rank.leaderboard([team('a', 4.0, 1.1), team('a', 4.0, 1.2)])


def test_rank_cli_reads_make_output_with_echo_lines(tmp_path, monkeypatch, capsys):
    for name, before, metric in (('alice', 6.0, 1.5), ('bob', 4.0, 1.1)):
        path = tmp_path / name / 'score.json'
        path.parent.mkdir()
        path.write_text('make: Entering directory\n' + json.dumps(team(name, before, metric)) + '\nmake: Leaving directory\n')
    monkeypatch.setattr(sys, 'argv', ['rank', '--report', str(tmp_path / 'alice/score.json'),
                                      '--report', f'bob={tmp_path / "bob/score.json"}'])
    assert rank.main() == 0
    board = json.loads(capsys.readouterr().out)
    assert [(t['team'], t['ranking_bonus'], t['total_points']) for t in board['teams']] == [('alice', 3, 9.0), ('bob', 2, 6.0)]


def _records(batch, compute_s, accuracy):
    record = {'Timing': {'Total': '20s', 'Key Generation': '1s', 'Encrypted model preprocessing': '2s',
                         'Encrypted computation': f'{compute_s:.6f}s'}}
    if batch > 1:
        record['Quality'] = {'Encrypted model quality': {'accuracy': accuracy, 'total_samples': batch},
                             'Harness model quality': {'accuracy': 0.97}}
    return [json.loads(json.dumps(record)) for _ in range(3)]


@pytest.mark.parametrize('slow_case,expected_tier,eligible', [(None, 3, True), ('C4', 0, False)])
def test_score_main_reports_the_policy_for_a_complete_legacy_run(tmp_path, monkeypatch, capsys, slow_case, expected_tier, eligible):
    """Six passing cases, each 25% faster than its baseline, except an optional 4% one."""
    base_rate = 50.0
    cases = {c['id']: c for c in score.CONFIG['cases']}

    def read_case(root, instance, runs, case_id, label):
        meta = json.loads((SUITE / 'results' / f'{case_id}_baseline' / 'case.json').read_text())
        ratio = 1.04 if case_id == slow_case else 1.25
        batch = cases[case_id]['batch']
        # the accuracy gate compares against the archived cleartext count of the same case
        accuracy = meta['verify']['own_cleartext']['cleartext_correct'] / batch
        return _records(batch, batch / (ratio * base_rate), accuracy), meta, tmp_path

    manifest = tmp_path / 'baseline.json'
    manifest.write_text(json.dumps({
        'suite_commit': score.CONFIG['suite']['commit'], 'seed': score.CONFIG['determinism']['seed'],
        'runs_per_case': score.CONFIG['determinism']['num_runs'], 'statistic': score.CONFIG['determinism']['statistic'],
        'cases': {cid: {'verified': 'PASS', 'images': cases[cid]['batch'], 'baseline_profile': 'legacy',
                        'throughput_images_per_s': base_rate} for cid in RANKED}}))
    monkeypatch.setattr(score, 'read_case', read_case)
    monkeypatch.setattr(sys, 'argv', ['score', '--evaluation-profile', 'legacy', '--baseline', str(manifest)])
    code = score.main()
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report['passed'] and report['policy'] == 'p1-10-point-v1', \
        {cid: case['reasons'] for cid, case in report['cases'].items()}
    assert report['correctness_points'] == 3.0 and report['cases_passed'] == 6
    assert report['improvement_points'] == expected_tier and report['ranking_eligible'] is eligible
    assert report['points_before_ranking_bonus'] == 3.0 + expected_tier
    assert all(case['points'] == 0.5 for case in report['cases'].values())
    # the synthetic timings are written with six decimals, which bounds the recovered ratio
    assert abs(report['improvement_min_ratio'] - (1.04 if slow_case else 1.25)) < 1e-6
    assert report['baseline_manifest_sha256'] == score.sha256(manifest)
    assert 'geometric_mean_throughput_ratio' not in report


def test_calibration_measured_on_another_chip_count_cannot_divide_a_ratio(tmp_path, monkeypatch, capsys):
    base_rate = 50.0
    cases = {c['id']: c for c in score.CONFIG['cases']}

    def read_case(root, instance, runs, case_id, label):
        meta = json.loads((SUITE / 'results' / f'{case_id}_baseline' / 'case.json').read_text())
        batch = cases[case_id]['batch']
        accuracy = meta['verify']['own_cleartext']['cleartext_correct'] / batch
        return _records(batch, batch / (1.25 * base_rate), accuracy), meta, tmp_path

    manifest = tmp_path / 'baseline.json'
    entries = {cid: {'verified': 'PASS', 'images': cases[cid]['batch'], 'baseline_profile': 'legacy',
                     'throughput_images_per_s': base_rate,
                     'execution': {'devices': [f'TPU_{i}' for i in range(cases[cid]['devices'])]}} for cid in RANKED}
    entries['C4']['execution']['devices'] = [f'TPU_{i}' for i in range(8)]   # eight-chip calibration for a four-chip case
    manifest.write_text(json.dumps({
        'suite_commit': score.CONFIG['suite']['commit'], 'seed': score.CONFIG['determinism']['seed'],
        'runs_per_case': score.CONFIG['determinism']['num_runs'], 'statistic': score.CONFIG['determinism']['statistic'],
        'cases': entries}))
    monkeypatch.setattr(score, 'read_case', read_case)
    monkeypatch.setattr(sys, 'argv', ['score', '--evaluation-profile', 'legacy', '--baseline', str(manifest)])
    code = score.main()
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and not report['passed']
    assert report['cases']['C4']['passed'] == 0 and any('device count' in r for r in report['cases']['C4']['reasons'])
    assert report['cases']['C3']['passed'] == 1 and report['cases']['C3']['throughput_ratio'] == pytest.approx(1.25)
    assert report['correctness_points'] == 2.5 and report['improvement_points'] == 0
