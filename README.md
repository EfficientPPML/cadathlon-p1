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

The default student schedule evaluates 100 images on one and eight TPU chips
(C3/C4). The archived baseline completed in 257.56 seconds on its measured host;
this is historical evidence, not a timing guarantee for another machine.
Correctness, actual-ring 128-bit security, input/model identity, calibration
provenance and cleanup checks remain enabled.

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
Only P1 packaging, navigation and standalone entry documentation are adapted;
the HE arithmetic, baseline, runner gates and scoring code retain their source
bytes. The repository retains the upstream [MIT license](LICENSE) and the
embedded fonts' license notices.
