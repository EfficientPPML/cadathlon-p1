include ../../../common/Makefile.problem

.PHONY: baseline-test baseline-plan
baseline-test: all
	$(PYTHON) runner.py --case ../test/test1.in --baseline
baseline-plan: all
	$(PYTHON) inspect_plan.py

.PHONY: suite-check suite-install suite-baseline suite-test suite-score \
        suite-gates suite-stop suite-rank

# The gate self-test is pure arithmetic over synthetic summaries -- no
# accelerator and no suite checkout -- so it is safe to run with `all`.
all: suite-gates
SUITE_DIR := $(abspath suite)
MLINF_ROOT ?= $(HOME)/ml-inference
export MLINF_ROOT
export P3_SUITE_DIR := $(SUITE_DIR)

suite-check:
	$(PYTHON) suite/install_suite.py --check
suite-install:
	$(PYTHON) suite/install_suite.py --install
# One ranked case: SIZE selects the instance (0 single, 1 small, 2 medium),
# P3_DEVICES the chip count. P3_BASELINE=1 runs baseline.py instead of bench_opt.py.
SIZE ?= 0
P3_DEVICES ?= 1
# The competition path uses C3/C4. Select legacy explicitly for six cases.
P1_BASELINE_PROFILE ?= student
export P1_BASELINE_PROFILE
EVALUATION_PROFILE ?= $(P1_BASELINE_PROFILE)
# Empty: evaluate.py uses the schedule's budget_s from suite_config.json (300 s
# for student) or 300 s when the profile is uncapped (legacy). A value may only
# lower a configured cap; legacy needs e.g. EVALUATION_BUDGET_S=7200.
EVALUATION_BUDGET_S ?=
BASELINE_LABEL ?= $(if $(filter student,$(P1_BASELINE_PROFILE)),student-baseline,baseline)
CANDIDATE_LABEL ?= $(if $(filter student,$(P1_BASELINE_PROFILE)),student-candidate,candidate)
BASELINE_MANIFEST ?= $(SUITE_DIR)/baseline_manifest$(if $(filter student,$(P1_BASELINE_PROFILE)),_student).json
# The suite keys measurements by instance only, so two cases that share an
# instance overwrite each other; archive each run under its own case id.
suite-baseline: suite-install
	cd $(MLINF_ROOT) && P3_BASELINE=1 P3_DEVICES=$(P3_DEVICES) \
	  $(PYTHON) harness/run_submission.py $(SIZE) --seed 3 --num_runs 3
	$(PYTHON) suite/archive.py --instance $(SIZE) --devices $(P3_DEVICES) \
	  --label $(BASELINE_LABEL) --suite-root $(MLINF_ROOT)
suite-test: suite-install
	cd $(MLINF_ROOT) && P3_DEVICES=$(P3_DEVICES) \
	  $(PYTHON) harness/run_submission.py $(SIZE) --seed 3 --num_runs 3
	$(PYTHON) suite/archive.py --instance $(SIZE) --devices $(P3_DEVICES) \
	  --label $(CANDIDATE_LABEL) --suite-root $(MLINF_ROOT)
suite-gates:
	$(PYTHON) suite/test_gates.py
suite-score:
	$(PYTHON) suite/score.py --suite-root $(MLINF_ROOT) --label $(CANDIDATE_LABEL) --baseline $(BASELINE_MANIFEST) --evaluation-profile $(EVALUATION_PROFILE)
suite-stop:
	cd $(MLINF_ROOT) && $(PYTHON) submissions/mnist/server_stop.py || true
# Cross-team ranking bonus of policy p1-10-point-v1: pass each team's saved
# suite-score JSON as TEAM=path. Offline arithmetic, no accelerator.
RANK_REPORTS ?=
suite-rank:
	$(PYTHON) suite/rank.py $(foreach report,$(RANK_REPORTS),--report $(report))

.PHONY: suite-evaluate suite-evaluate-baseline
# Full selected schedule with wall-time budget, per-case archives and cleanup.
suite-evaluate:
	$(PYTHON) suite/evaluate.py --profile $(P1_BASELINE_PROFILE) $(if $(EVALUATION_BUDGET_S),--budget-s $(EVALUATION_BUDGET_S))
suite-evaluate-baseline:
	$(PYTHON) suite/evaluate.py --profile $(P1_BASELINE_PROFILE) --baseline $(if $(EVALUATION_BUDGET_S),--budget-s $(EVALUATION_BUDGET_S))

.PHONY: suite-evaluate-hidden suite-evaluate-baseline-hidden suite-score-hidden suite-overall
# Organizer-run hidden schedule (student profile: C5/C6, 1000 images on one and
# four chips, same ring/model/seed). In ONE judge checkout, calibrate both
# schedules once (suite-evaluate-baseline, suite-evaluate-baseline-hidden);
# the scorer finds calibration proofs only under this checkout's
# suite/calibration/. Then, one submission at a time, install it, run BOTH
# suite-evaluate (public C3/C4) and suite-evaluate-hidden, and save the output
# of suite-overall, which scores both schedules under policy p1-10-point-v1.
# The hidden budget defaults to hidden_budget_s in suite_config.json;
# HIDDEN_BUDGET_S may only lower it.
HIDDEN_BUDGET_S ?=
HIDDEN_CANDIDATE_LABEL ?= $(P1_BASELINE_PROFILE)-hidden-candidate
HIDDEN_BASELINE_MANIFEST ?= $(SUITE_DIR)/baseline_manifest_$(P1_BASELINE_PROFILE)_hidden.json
suite-evaluate-hidden:
	$(PYTHON) suite/evaluate.py --profile $(P1_BASELINE_PROFILE) --hidden $(if $(HIDDEN_BUDGET_S),--budget-s $(HIDDEN_BUDGET_S))
suite-evaluate-baseline-hidden:
	$(PYTHON) suite/evaluate.py --profile $(P1_BASELINE_PROFILE) --hidden --baseline $(if $(HIDDEN_BUDGET_S),--budget-s $(HIDDEN_BUDGET_S))
suite-score-hidden:
	$(PYTHON) suite/score.py --suite-root $(MLINF_ROOT) --hidden --label $(HIDDEN_CANDIDATE_LABEL) --baseline $(HIDDEN_BASELINE_MANIFEST) --evaluation-profile $(EVALUATION_PROFILE)
suite-overall:
	$(PYTHON) suite/overall.py --suite-root $(MLINF_ROOT) --evaluation-profile $(EVALUATION_PROFILE) \
	  --public-label $(CANDIDATE_LABEL) --public-baseline $(BASELINE_MANIFEST) \
	  --hidden-label $(HIDDEN_CANDIDATE_LABEL) --hidden-baseline $(HIDDEN_BASELINE_MANIFEST)
