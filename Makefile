PYTHON ?= python3
P1_SRC := cadathlon/non_agent_pset/p1_benchmark/src

.PHONY: all reference-test check docs baseline-plan submit suite-check suite-install suite-evaluate-baseline suite-evaluate suite-score suite-rank suite-stop clean \
        suite-evaluate-hidden suite-evaluate-baseline-hidden suite-score-hidden suite-overall

all:
	$(MAKE) -C $(P1_SRC) all PYTHON="$(PYTHON)"

reference-test:
	JAX_PLATFORMS=cpu $(MAKE) -C $(P1_SRC) reference-test PYTHON="$(PYTHON)"

check: reference-test
	JAX_PLATFORMS=cpu $(PYTHON) -m pytest -q cadathlon/tests

docs:
	$(PYTHON) cadathlon/common/render_docs.py

# suite-rank forwards RANK_REPORTS; pass absolute report paths because the inner make runs from src/.
# The *-hidden targets and suite-overall are the organizer's hidden-schedule (C5/C6) path.
baseline-plan submit suite-check suite-install suite-evaluate-baseline suite-evaluate suite-score suite-rank suite-stop clean \
suite-evaluate-hidden suite-evaluate-baseline-hidden suite-score-hidden suite-overall:
	$(MAKE) -C $(P1_SRC) $@ PYTHON="$(PYTHON)"
