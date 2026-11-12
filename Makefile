PYTHON ?= python3
P1_SRC := cadathlon/non_agent_pset/p1_benchmark/src
# P5's pinned WISE planner: `make install-wise` clones it to $(WISE_ROOT), by
# default ../WISE next to this checkout. `make check` needs it for P5's reference
# check and exports it (as an absolute path) for the WISE planner tests, which
# skip only when WISE_ROOT is unset, e.g. when pytest runs on its own.
WISE_DEFAULT := $(abspath ../WISE)
ifeq ($(strip $(WISE_ROOT)),)
override WISE_ROOT := $(wildcard $(WISE_DEFAULT))
endif
ifneq ($(WISE_ROOT),)
override WISE_ROOT := $(abspath $(WISE_ROOT))
export WISE_ROOT
else
unexport WISE_ROOT
endif
# Nested makes take WISE_ROOT from the environment above, not as the original
# (possibly relative) command-line value that MAKEFLAGS would pass down.
MAKEOVERRIDES := $(filter-out WISE_ROOT=%,$(MAKEOVERRIDES))

.PHONY: all reference-test check docs release clean install-wise baseline-plan submit suite-check suite-install suite-evaluate-baseline suite-evaluate suite-score suite-rank suite-stop \
        suite-evaluate-hidden suite-evaluate-baseline-hidden suite-score-hidden suite-overall

# Whole problem set (cadathlon/Makefile): syntax/package checks and the public
# plaintext references of all five problems. P5's references need the pinned
# WISE planner: make install-wise
all:
	$(MAKE) -C cadathlon all PYTHON="$(PYTHON)"

reference-test:
	JAX_PLATFORMS=cpu $(MAKE) -C cadathlon reference-test PYTHON="$(PYTHON)"

check: reference-test
	JAX_PLATFORMS=cpu $(PYTHON) -m pytest -q cadathlon/tests

docs:
	$(PYTHON) cadathlon/common/render_docs.py

release:
	$(MAKE) -C cadathlon release PYTHON="$(PYTHON)"

install-wise:
	$(MAKE) -C cadathlon/agent_pset/p5_wise/src install-wise WISE_ROOT="$(or $(WISE_ROOT),$(WISE_DEFAULT))" PYTHON="$(PYTHON)"

clean:
	$(MAKE) -C cadathlon clean PYTHON="$(PYTHON)"

# P1 targets. suite-rank forwards RANK_REPORTS; pass absolute report paths because the inner make runs from src/.
# The *-hidden targets and suite-overall are the organizer's hidden-schedule (C5/C6) path.
baseline-plan submit suite-check suite-install suite-evaluate-baseline suite-evaluate suite-score suite-rank suite-stop \
suite-evaluate-hidden suite-evaluate-baseline-hidden suite-score-hidden suite-overall:
	$(MAKE) -C $(P1_SRC) $@ PYTHON="$(PYTHON)"
