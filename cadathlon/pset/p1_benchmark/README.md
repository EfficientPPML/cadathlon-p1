# Problem 1: benchmark track

Read [the handout](doc/problem.html) or [the PDF](doc/problem.pdf).
Problem 1 is solved without AI assistance (no generative AI or coding agents),
like Problem 3, with which it shares one leaderboard.
The default competition schedule is **student**: C3/C4, 100 images on one/four
TPU chips. Implement `src/bench_opt.py`, beginning with `baseline_student.py`
and its supplied-key handling. The explicit legacy profile ranks six cases;
the six local synthetic fixtures remain regression checks for both schedules.

After organizers stage the upstream suite, data and cleartext checkpoint:

```bash
make -C src suite-evaluate-baseline  # full TPU schedule; publishes calibration
# Implement bench_opt.py using the student reference's policy and keys.
make -C src suite-evaluate           # full TPU candidate schedule
make -C src suite-score              # offline scoring; does not run inference
```

All three commands select `student` by default. Both full evaluations enforce
300 seconds including setup, repetitions, archiving and cleanup. The scorer
requires the matching completed report and exact archived measurements; a
failed run, changed archives, wrong profile or mismatched model/data/hardware/
software cannot borrow a previous success. Ranked weight bytes must match
`weights/MANIFEST.json`.

**Hidden cases.** C5 and C6 (1000 images on one and four chips, same student
ring, frozen model and seed) are organizer-run hidden cases of the student
schedule. Contestants do not run them before judging and their calibration
(`baseline_manifest_student_hidden.json` and its archives, excluded from the
contestant release and from git) is not published. The organizer works in one
judge checkout on the contest host, because the scorer finds calibration
proofs only under that checkout's `src/suite/calibration/`:

```bash
# once, in the judge checkout
make -C src suite-evaluate-baseline          # public C3/C4 calibration
make -C src suite-evaluate-baseline-hidden   # hidden C5/C6 calibration (1800 s budget)
# per submission: install its bench_opt.py, helper modules and cross.patch, then
make -C src suite-evaluate                   # public C3/C4, rerun by the judge
make -C src suite-evaluate-hidden            # hidden C5/C6
make -C src suite-overall > /abs/teams/<team>/overall.json
```

`suite-overall` (`src/suite/overall.py`) scores both schedules itself, shares
the 3 correctness points over all four cases (0.75 each), decides the
improvement tier by the smallest of the four ratios and the ranking metric by
their geometric mean, and refuses to combine public and hidden runs that do not
come from the same submission sources, model and software. A missing,
interrupted or failed hidden run fails C5/C6 but not C3/C4; because improvement
points need every case to pass, the submission then keeps only the 1.5 public
correctness points and is not ranked. The next submission's evaluations replace
the reports, so save each overall report first. `make -C src suite-score-hidden`
shows the hidden verdict alone, and `make -C src suite-rank` ranks the saved
overall reports across teams.

For local development only, from `src/`, run `make reference-test`, `make baseline-plan`,
`make baseline-test`, then `make test` after implementing the TODO.
`make baseline-plan` prints the actual derived plan and the slot occupancy the
problem asks you to improve. The HE commands may take minutes and gigabytes.
Set `CROSS_ROOT` if this package is extracted outside the repository.
`make submit` makes a local archive. Public synthetic cases verify arithmetic;
official accuracy runs use organizer-supplied trained weights and images, as
specified in the handout.

For experimental local iteration, `python runner.py --student-baseline --case
../test/test1.in --warmups 1 --repeats 3` selects `baseline_student.py`. It uses
the same model, images and numerical gates, with a derived 128-bit-secure
N=8192 ring, 42-bit composite scale and four HYBRID digits. The local
`runner.py --baseline` still selects the legacy smoke reference. The suite
Make targets default to the student ring for both key generation and evaluation.
A single-case development command is, for example:

```bash
make -C src suite-baseline SIZE=1 P3_DEVICES=1 P1_BASELINE_PROFILE=student
```

To time both student cases through the actual harness, including setup,
three repetitions, archiving and cleanup:

```bash
python src/suite/evaluate.py --profile student --baseline --budget-s 300
python src/suite/evaluate.py --profile student --budget-s 300
make -C src suite-score P1_BASELINE_PROFILE=student
```

Stage the upstream data and cleartext checkpoint first. The command rejects a
CPU/GPU fallback and returns nonzero on timeout, failed correctness or cleanup.
Measured on the four-chip TPU v6e host (`v6e-4`), the complete baseline command
finished in **276.63 seconds (4m37s)**, with both cases passing all gates and
100% agreement with the same model in cleartext (C3: 53.97 images/s on one
chip; C4: 130.21 images/s on four chips). See
`../../validation/p1_student_suite.json` for scope and environment; the earlier
eight-chip run (257.47 s, C4 on eight chips) is kept under
`../../validation/p1_student_v6e8/`. Recalibrate on contest hardware.
Scoring follows the 10-point policy in the handout's section 4: 3 correctness
points shared by the ranked cases, 0–4 improvement points from the smallest
throughput ratio over all cases (at least 5/10/20/40% faster everywhere), and a
3/2/1 ranking bonus for the three best eligible teams (`make -C src suite-rank`).
Under the short profile each passing case is worth 1.5 correctness points.

This uses separate `C3_student-baseline`/`C4_student-baseline` archives and needs
a measured `src/suite/baseline_manifest_student.json` for improvement points.
The scorer refuses to compare different profiles. Candidate services using
this profile must accept its supplied keys. A smaller ring alone does not
prove the five-minute evaluation target; use the short profile for the 1.5-hour slot after the
organizer reproduces and calibrates the selected workload on contest hardware.

New `student-v2` calibrations bind a complete evaluation ID to public aggregate
reports and measurement hashes in `src/suite/calibration/student-baseline/`.
That proof is included in the release and survives `make clean`; private
labels, images, predictions and keys are not copied into it. A failed later
calibration preserves the published successful proof. Candidate reports stay
under `src/suite/run/evaluation-student-candidate/`; `make clean` removes them,
so rerun before scoring. The recorded 276.63-second four-chip run includes this
complete proof and passed baseline self-scoring (3.0 of 10 points: full
correctness, no improvement points). The earlier eight-chip 257.47-second proof and the 262.14-second
pre-v2 evidence are preserved separately under `../../validation/p1_student_v6e8/`
and `../../validation/p1_student_pre_v2/historical_run.json`; neither matches
the four-chip schedule, so neither can replace the new proof. The shipped proof
predates the hidden cases: it is bound to the earlier `suite_config.json`, so
`suite-score` rejects it until `make -C src suite-evaluate-baseline` has been
run again on the contest host.

Use `P1_BASELINE_PROFILE=legacy` explicitly with `suite-baseline`, `suite-test`
or `suite-score` for the old schedule. To run the entire longer legacy schedule,
use `make -C src suite-evaluate P1_BASELINE_PROFILE=legacy EVALUATION_BUDGET_S=7200`.
The larger budget is accepted only for legacy evaluations. Plaintext fixture
checks and `suite-gates` are CPU checks, not evidence of TPU performance.

The local runner requires the device count declared by each fixture; it fails
before setup if fewer devices are available. CPU development can emulate four
devices with `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4`.
Such runs check arithmetic and batching, not TPU performance. Reports include
the actual execution rings checked by the security gate.

The local runner's `setup_s` times only the service constructor. The baseline
initializes keys and compiles lazily on its first request. `warmup_s` and
`warmup_server_s` retain the excluded warmup samples; with `--warmups 0`,
`cold_start_included` is true and the first measured request includes any
lazy setup. `total_s` includes fixture loading, setup, all requests and service
release, so use it when planning an evaluation slot. These smoke timings do
not replace the ranked suite's timing boundaries.

Suite stage 3 accepts an optional `service.warmup_input()` hook returning a
dummy ciphertext bundle that covers every physical execution shape used by
that service. The baseline encrypts one zero image and repeats its ciphertext
across one complete device lane group: every later group runs the same
Mapping. This avoids encrypting and evaluating the entire 1000-image batch
just to warm that Mapping. Both input and output security checks still run.
Services without the hook retain full-batch warmup, and ranked input counts
and timing boundaries are unchanged. This reduces warmup work; it does not
establish that the full six-case evaluation meets a five-minute budget.
