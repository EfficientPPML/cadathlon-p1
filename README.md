# CADathlon P1 — encrypted MNIST throughput

P1's starter implementation, real CROSS baseline, six public development
fixtures, ranked evaluation tools, scoring and handout are collected here with
their CROSS runtime dependencies. No separate CROSS checkout is required.

Start with the [problem handout](cadathlon/non_agent_pset/p1_benchmark/doc/problem.md)
([HTML](cadathlon/non_agent_pset/p1_benchmark/doc/problem.html),
[PDF](cadathlon/non_agent_pset/p1_benchmark/doc/problem.pdf)) and implement
[bench_opt.py](cadathlon/non_agent_pset/p1_benchmark/src/bench_opt.py).
The [P1 guide](cadathlon/non_agent_pset/p1_benchmark/README.md) describes the
interfaces and evaluation rules. The original directory layout is retained so
runner, common-library and suite-stage imports keep their established paths.

## Local checks

Python 3.13 is the tested interpreter. Install the runtime and test dependencies
in a virtual environment:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
make check
```

`make check` runs the six plaintext fixture checks and the focused P1 CPU
regressions. It does not run a TPU benchmark. The recorded package-test
environment and results are in [validation/](validation/).

For a CPU-only PyTorch installation, use the official CPU package index when
installing `torch`: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
For TPU evaluation, install the TPU JAX dependencies (`python -m pip install
'jax[tpu]'`) in the evaluation environment, then reproduce the baseline on the
contest hardware. The benchmark rejects a CPU/GPU fallback.

## Ranked evaluation

The default student schedule evaluates 100 images on one and four TPU chips
(C3/C4), the shape of a four-chip TPU v6e host (`v6e-4`). The shipped student
calibration was measured on that host on 2026-11-07: the complete schedule took
276.63 seconds of its 300-second budget, with C3 at 53.97 images/s on one chip
and C4 at 130.21 images/s on four chips. This is evidence for that machine, not
a timing guarantee for another one. Correctness, actual-ring 128-bit security,
input/model identity, calibration provenance and cleanup checks remain enabled.

## Scoring policy (10 points)

Problem 1 contributes at most **10 points** to the event, computed by policy
`p1-10-point-v1` (handout section 4) in three tiers over the six ranked cases:

| Tier | Points | Rule |
|---|---:|---|
| Correctness | 3 | shared equally by the ranked cases: 0.5 per passing case on the six-case schedule (1.5 per case when the two-case student schedule is scored) |
| Improvement | 0–4 | all cases pass and the *smallest* throughput ratio against the calibrated baseline is ≥ 1.05 / 1.10 / 1.20 / 1.40 for 1 / 2 / 3 / 4 points |
| Ranking bonus | 0–3 | the three best teams with at least one improvement point, by geometric mean of their ratios: 3, 2, 1 points; equal rounded metrics share the bonus |

`make suite-score` prints a team's correctness and improvement points plus its
ranking metric; `make suite-rank RANK_REPORTS="alice=$PWD/teams/alice.json bob=$PWD/teams/bob.json"`
(absolute paths; the target forwards to `src/`) assigns the bonus across the
saved reports and refuses to mix schedules, policies or calibration manifests. An unmodified baseline scores 3.0. The
report (`REPORT.md`) carries no separate points but is required for the judge's
privacy, parameter and memory audit.

The ranked evaluation uses the separately maintained upstream harness and MNIST
data. Its pinned revision is in
[suite_config.json](cadathlon/non_agent_pset/p1_benchmark/src/suite/suite_config.json).
The ranked model checkpoint and its hash manifest are included under
[weights/](cadathlon/non_agent_pset/p1_benchmark/weights/).

```bash
git clone https://github.com/fhe-benchmarking/ml-inference "$HOME/ml-inference"
git -C "$HOME/ml-inference" checkout 739378698fc1f668aff7246724b45cff76f04a98
export MLINF_ROOT="$HOME/ml-inference"
# Stage the MNIST data as described by that harness and the P1 handout.
make suite-check
make suite-evaluate-baseline
# Implement bench_opt.py, then:
make suite-evaluate
make suite-score
```

Use `make baseline-plan` to inspect the packing plan without a full evaluation.
For local encrypted smoke checks, use the commands in the P1 guide. An external
`CROSS_ROOT` is optional; by default the bundled runtime is used.

## Verified on the four-chip host (2026-11-07)

Host: GCE `ct6e-standard-4t-tpu` (TPU `v6e-4`, four "TPU v6 lite" chips),
Python 3.13.18, jax/jaxlib 0.12.0.1, libtpu 0.0.50.1, numpy 2.5.4,
torch 2.15.0+cpu; see [validation/environment.json](validation/environment.json)
and [validation/report.json](validation/report.json). The following tasks work
end to end on this host:

- `make check` (CPU): 6/6 plaintext fixtures exact, 23/23 scorer gate
  behaviors, 183/183 pytest tests ([validation/check.log](validation/check.log)).
- `make baseline-plan` and `inspect_plan.py --student-baseline` (CPU): legacy
  plan N=16384, 8192 slots, 9.57% occupancy, log2(QP)=365.0; student plan
  N=8192, 4096 slots, 19.14% occupancy, log2(QP)=212.46; both at 128-bit.
- `make docs`: the HTML and 12-page PDF handouts re-render from `problem.md`.
- Local encrypted smoke checks on the TPU (`runner.py`): all six fixtures with
  `--student-baseline` on 1, 2 and 4 chips, and test1/test4/test6 with the
  legacy `--baseline`; every logit, argmax and actual-ring security gate passed
  ([validation/tpu_smoke_v6e4.json](validation/tpu_smoke_v6e4.json)).
- `make suite-check` and `make suite-install`: the pinned `ml-inference`
  harness is byte-identical to its commit and the eight stage shims install.
- `make suite-evaluate-baseline` (student, C3/C4): the complete schedule
  finished in 276.63 s of the 300 s budget, 96/100 correct in both cases,
  100% agreement with the same model in cleartext, calibration published
  ([cadathlon/validation/p1_student_suite.json](cadathlon/validation/p1_student_suite.json)).
- `make suite-score CANDIDATE_LABEL=student-baseline`: baseline self-score of
  3.0 of 10 points under policy `p1-10-point-v1` (full correctness, no
  improvement points, not eligible for the ranking bonus).
- `make suite-evaluate` with the unimplemented `bench_opt.py`: stops in stage 3
  with `NotImplementedError` after 20.7 s, cleanup passes and `make suite-score`
  awards 0 points. That is the designed state until a contestant implements it.
- `make submit` (31 files) and the contestant release (125 files) package
  without run directories, per-image evidence or keys.
- `make suite-evaluate-baseline P1_BASELINE_PROFILE=legacy EVALUATION_BUDGET_S=7200`
  (legacy, C1–C6): all six cases pass in 1685.94 s, every case verified with
  100% cleartext agreement; the republished
  [baseline_manifest.json](cadathlon/non_agent_pset/p1_benchmark/src/suite/baseline_manifest.json)
  now holds four-chip numbers and the legacy self-score is 3.0 of 10 points
  (0.5 per passing case, no improvement points;
  [cadathlon/validation/p1_legacy_suite.json](cadathlon/validation/p1_legacy_suite.json)).

Ranked student schedule on four chips (median of three harness runs; the same
100 images and frozen model as the earlier eight-chip calibration):

| Case | Images | Chips | Stage 7 compute | Images/s | ms/image | Server execute | One-time setup | Correct | Cleartext agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 | 100 | 1 | 1.8528 s | 53.97 | 18.53 | 1.5315 s | 75.5 s | 96/100 | 100% |
| C4 | 100 | 4 | 0.7680 s | 130.21 | 7.68 | 0.4479 s | 80.5 s | 96/100 | 100% |

On the earlier eight-chip host the same images took 1.8138 s for C3 and
0.5757 s for C4 with eight chips. Throughput ratios are only meaningful against
a calibration measured on the same host and schedule.

Legacy six-case schedule on four chips (`baseline.py`, N=16384 ring, median of
three harness runs):

| Case | Images | Chips | Stage 7 compute | Images/s | ms/image | Server execute | Server ms/image | One-time setup | Correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1 | 1 | 0.1193 s | 8.38 | 119.30 | 0.0284 s | 28.41 | 94.4 s | 1/1 |
| C2 | 1 | 4 | 0.1221 s | 8.19 | 122.10 | 0.0306 s | 30.56 | 98.4 s | 1/1 |
| C3 | 100 | 1 | 3.0700 s | 32.57 | 30.70 | 2.6976 s | 26.98 | 94.9 s | 96/100 |
| C4 | 100 | 4 | 1.1557 s | 86.53 | 11.56 | 0.7533 s | 7.53 | 98.4 s | 96/100 |
| C5 | 1000 | 1 | 30.0190 s | 33.31 | 30.02 | 26.9654 s | 26.97 | 95.4 s | 976/1000 |
| C6 | 1000 | 4 | 10.7333 s | 93.17 | 10.73 | 7.5094 s | 7.51 | 98.4 s | 976/1000 |

On one chip the server cost is flat at about 27 ms per image from batch 100 to
1000; four chips give 3.59× at batch 1000, and at batch 1 four chips are slightly
slower than one (30.6 ms against 28.4 ms).

Local-runner smoke timings on the TPU (development diagnostics, two warmups and
five measured requests; not the ranked metric):

| Fixture | Batch | Devices | Ring | End-to-end median | Server execute median | Server ms/image | Max abs logit error |
|---|---:|---:|---|---:|---:|---:|---:|
| test1 | 1 | 1 | student N=8192 | 72.2 ms | 16.8 ms | 16.8 ms | 8.4e-09 |
| test2 | 2 | 1 | student N=8192 | 137.9 ms | 32.0 ms | 16.0 ms | 5.7e-09 |
| test3 | 4 | 1 | student N=8192 | 280.4 ms | 62.6 ms | 15.6 ms | 1.1e-08 |
| test4 | 4 | 2 | student N=8192 | 253.1 ms | 36.0 ms | 9.0 ms | 1.4e-08 |
| test5 | 8 | 2 | student N=8192 | 501.0 ms | 69.2 ms | 8.6 ms | 1.3e-08 |
| test6 | 8 | 4 | student N=8192 | 472.4 ms | 38.3 ms | 4.8 ms | 1.3e-08 |
| test1 | 1 | 1 | legacy N=16384 | 142.4 ms | 28.9 ms | 28.9 ms | 2.4e-13 |
| test4 | 4 | 2 | legacy N=16384 | 509.8 ms | 60.4 ms | 15.1 ms | 2.6e-13 |
| test6 | 8 | 4 | legacy N=16384 | 957.2 ms | 63.1 ms | 7.9 ms | 1.8e-13 |

The first request of each run pays key generation and compilation (about 66 s
on one chip and about 72 s on two or four chips for the student ring, and about
85 s on one chip to 90 s on four chips for the legacy ring) and is excluded
from the medians.

Resolved follow-up from the review of this migration: the legacy scoring path in
[score.py](cadathlon/non_agent_pset/p1_benchmark/src/suite/score.py) never binds
a calibration entry's device count to the case it scores, so a legacy manifest
measured with a different chip count would be divided silently; the student path
checks `device_kinds`. The republished four-chip `baseline_manifest.json`
removed the immediate hazard, and the 10-point policy change closed the gap:
`score.py` now fails a case whose calibration entry records a different device
count, and `suite-rank` refuses reports scored against different manifests.

## Submission and documentation

Complete the problem's `REPORT.md` and `cross.patch`, then run `make submit`.
The submission archive contains P1 and its shared runner utilities; the judge
applies `cross.patch` to the declared CROSS runtime. It excludes ephemeral runs,
keys, organizer per-image evidence and golden files.

To rebuild the offline HTML/PDF handout:

```bash
python -m pip install -r cadathlon/requirements-docs.txt
make docs
```

## Provenance

Extracted from `EfficientPPML/CROSS` commit
`3a49b207acc851a8d599b9b3f3053b8af1d199aa`. The initial copied-file hashes are in
[upstream-source-manifest.json](validation/upstream-source-manifest.json).
Only P1 packaging, navigation and standalone entry documentation are adapted,
and the scorer carries the 10-point event policy (`score.py`, `rank.py`); the
HE arithmetic, baseline and runner gates retain their source bytes. The repository retains the upstream [MIT license](LICENSE) and the
embedded fonts' license notices.
