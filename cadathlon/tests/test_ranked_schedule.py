"""The ranked schedule is bound to the four-chip TPU v6e host (v6e-4)."""
from pathlib import Path
import json

PROBLEM = Path(__file__).resolve().parents[1] / 'non_agent_pset/p1_benchmark'
CONFIG = json.loads((PROBLEM / 'src/suite/suite_config.json').read_text())
CASES = {case['id']: case for case in CONFIG['cases']}


def test_student_schedule_is_c3_c4_on_one_and_four_chips():
    assert CONFIG['evaluation_profiles']['student']['case_ids'] == ['C3', 'C4']
    assert (CASES['C3']['batch'], CASES['C3']['devices']) == (100, 1)
    assert (CASES['C4']['batch'], CASES['C4']['devices']) == (100, 4)


def test_every_ranked_case_fits_the_four_chip_host():
    # The baseline pads a short window across its lanes, so C2 (one image on
    # four chips) is legitimate; only the chip count itself is bound here.
    assert {case['devices'] for case in CASES.values()} == {1, 4}


def test_public_fixtures_never_need_more_devices_than_the_ranked_schedule():
    fixtures = [json.loads(path.read_text()) for path in sorted((PROBLEM / 'test').glob('test*.in'))]
    assert fixtures and max(case['devices'] for case in fixtures) <= max(case['devices'] for case in CASES.values())
