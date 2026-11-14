# Sources and decisions

Initially inspected on 2026-09-19; BERT sources added on 2026-09-21.

| Source | What the problem set uses |
|---|---|
| `/home/jianming_gatech/CADathon_Template/Example 1/README` and `doc/p2-physical.html` | Package layout, TODO files, goldens, six or more cases, runtime limits, five-section handout |
| `/home/jianming_gatech/CADathon_Template/Example_2_2019_p6/doc/p6.html` | Neural-network optimization framing, runnable starter and explicit scoring |
| `/home/jianming_gatech/CROSS_paper.pdf`, Tong et al., *Leveraging ASIC AI Chips for Homomorphic Encryption*, HPCA 2026; [online paper](https://arxiv.org/abs/2501.07047) | CKKS accelerator motivation, BAT and MAT; §§III–IV and evaluation in §V. Historical kernel/ML results are not contest baselines. |
| `/home/jianming_gatech/CROSS_Tutorial/{beginner,cross,tpu,ntt,open_challenge}.html` and `README.md`; [tutorial](https://efficientppml.github.io/CROSS_Tutorial/) | HE background, profiling, NTT/BConv, scheduling and layout optimization |
| `jaxite_word/API_REFERENCE.md`, `nn.py`, `packing.py`, `mapping.py`, `he_params.py` | Canonical rank-5 Polynomial, secure ring derivation, lazy constants, frontend limitations, bootstrap barriers |
| `demos/{canonical_demo,encrypted_demos,lenet_train,alexnet_infer,prepare_demo_assets}.py` | Model definitions, weight loading, AlexNet client pooling, baseline implementation |
| [Official Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B) and [pinned config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json) | Model identity, tensor dimensions, RMSNorm/RoPE settings, tied embeddings, tokenizer mode |
| [FHE BERT benchmark, pinned commit](https://github.com/fhe-benchmarking/BERT/tree/170bfe567545d74d0fad785a052518351d37bc93) | Unmodified harness and seven submission stages; MRPC counts 1/100/200/408; 75% quality threshold; outer stage timings; output-check and repeated-seed limitations |
| [Pinned BERT checkpoint](https://huggingface.co/google-bert/bert-base-cased-finetuned-mrpc/tree/f150c1d609d1e50dd5e2e5408661cfac8339277c) | Cased tokenizer and BERT-Base dimensions, LayerNorm epsilon and model identity; upstream explicitly loads BertForSequenceClassification |
| [Transformers 4.51.3 BERT implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/bert/modeling_bert.py) | Independent synthetic validation of embeddings, bidirectional attention, post-LayerNorm blocks, pooler and classifier |

The Qwen repository revision is pinned in `agent_pset/p4_qwen/test/model_source.json`.
The bundled config is unmodified; model weights/tokenizer files must be staged
from that same revision by organizers. Do not resolve `main` again on contest day.

The NumPy decoder oracle was checked against the
[Transformers 4.51.3 Qwen3 implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py),
including Q/K per-head normalization, half-vector RoPE rotation and KV-head
repetition. This is a semantics check on the small public cases, not a
full-checkpoint or encrypted-inference result.

The full Qwen target is beyond CROSS's current automatic CNN frontend. In
particular, replacing its SiLU with the frontend's default square is not an
equivalent Qwen implementation. The assignment allows calibrated polynomial
approximations, with explicit quality gates against the original checkpoint.
It requires encrypted attention and encrypted depth refresh as necessary.

BERT's upstream DESILO implementation informs the allowed client embedding
boundary and public attention-mask leakage, but is not copied as the CROSS
backend. `agent_pset/p2_bert/src/benchmark.py` pins all eleven harness Python files by
SHA-256. Staging retains their Apache license and leaves their contents
unchanged. The upstream `run_submission.py` sets no model revision itself;
organizers must provision a pinned offline cache for its cleartext reference.
The BERT NumPy oracle uses synthetic weights and is explicitly plaintext.
Its six cases were independently compared with Transformers; they do not
establish trained MRPC accuracy or encrypted BERT feasibility.

Security target, workload sizes, tolerances, time limits, point allocations,
speedup caps, AI-use policy and contest duration are design choices in this
problem set, not claims made by the paper or measured facts. Organizers must
validate and freeze these choices before release as described in ORGANIZER.md.

## Repository migration (2026-11-10)

This problem set moved from `EfficientPPML/CROSS` into this repository, which
already held the standalone P1. Where each part comes from:

| Part | Source |
|---|---|
| P1 (`non_agent_pset/p1_benchmark`), its tests that differ from CROSS or are new here (`test_evaluation_schedule.py`, `test_scoring.py`, `test_student_evidence.py`, `test_hidden_schedule.py`, `test_ranked_schedule.py`, `test_scoring_policy.py`, `test_standalone_package.py`) and its validation records (`p1_student_suite.json`, `p1_legacy_suite.json`, `p1_student_v6e8/`, `p1_student_pre_v2/`) | This repository's own P1: extracted from CROSS `3a49b20` on 2026-11-06 and developed here since (four-chip schedule, 10-point policy, hidden C5/C6 cases) |
| P2, P3, P4, `common/`, the other tests, `README.md`, `ORGANIZER.md`, `VALIDATION.md` and the other top-level `validation/` records | CROSS branch `fix/competition-readiness`, commit `3a49b207acc851a8d599b9b3f3053b8af1d199aa` |
| P5 (`agent_pset/p5_wise`) and its `test_wise_*` tests | CROSS branch `feat/p5-wise-tpu-20261105`, commit `3cec9fa`, replayed onto `3a49b20` as CROSS commit `152fb80b1d56d0bf7fec0f61f5c5bfd853f59c49` (branch `migrate-cadathlon`); archived here as `6613bad` and `461bb8e` (see below) |
| CROSS runtime: `jaxite_word/`, `demos/`, `profile_analysis/`, `pytest.ini` (repository root) | CROSS `3a49b20`; `jaxite_word/` is identical in `3a49b20` and `3cec9fa` |

Replaying P5 onto `3a49b20` conflicted in `common/package.py` (its
runtime-file lists, submission and release contents, and problem list), in the
problem list of `common/render_docs.py`, in `README.md` and the root
`.gitignore`, and in three tests that `3a49b20` had extended with further
cases (its versions were kept).

Changes made here on top of those sources:

- `common/package.py` and `common/render_docs.py` serve all five problems. Each
  submission keeps its line's contents: P1 ships only the shared tooling, P5
  the WISE runtime subset, and the other problems the complete shared runtime;
  `submit` also adds any other shared module a submission's sources import by
  an import statement (it scans only regular files, archives a source that
  does not parse without following its imports, and treats an imported shared
  module that is a symlink as an error). `render_docs.py` strips only trailing
  spaces and tabs, outside `<pre>` blocks, from the rendered HTML. The package tests (`test_package.py`,
  and the release fixtures of `test_standalone_package.py` and
  `test_hidden_schedule.py`) follow suit, and new tests check that the narrowed
  P1 and P5 runtimes contain every shared module their sources import,
  directly or indirectly, that a new import reaches the archive, that editor
  lock links are skipped, that unparsable sources are still archived, and that
  the release's runtime covers every problem's imports.
- `test_cnn_evaluation.py` stubs the package-version lookup in one test, as
  `test_evaluation_schedule.py` already did, so it passes on hosts without `libtpu`.
- P5's `EXCLUDE` also drops golden files, hidden or full-reference test cases
  (as P1–P4's do) and the native OpenFHE/WISE build directories its README
  creates, and it no longer drops the empty `cross.patch` template, which
  `make submit` requires and P1–P4 ship. Every problem's `EXCLUDE` drops raw
  `results*.json` copied into the problem directory for `make submit`.
- Handout text (`doc/problem.md`, and the READMEs of P2 and P5 where they say
  the same): P1 names the right record for its legacy figures; P2 has
  contestants patch the CROSS runtime at this repository's root (which the
  extracted release does not include) and point `CROSS_ROOT` there; P3 gives P1's 10 points and no longer claims
  that its 20-point bonus cannot outweigh P1; P1, P3, P4 and P5 say that
  `make submit` packages `results*.json` from the problem directory; P5 tells
  users of the extracted release to set `CROSS_ROOT`. All five
  handouts were re-rendered to HTML and PDF with the shared five-problem
  navigation (PDFs on Pango 1.52, which sets some typography differently).
- The set-level documents (`README.md`, `ORGANIZER.md`, the `VALIDATION.md`
  header) and P5's README, runbook and status note describe this repository and
  the current P1; `ORGANIZER.md` and `README.md` give the legacy P1
  recalibration its 7200-second budget. P5's status note says that its raw
  evidence directory was the author's local one and was never committed.
- The root `Makefile` builds, checks, renders, releases and cleans the whole
  set through `cadathlon/Makefile` and keeps P1's targets; its new
  `install-wise` clones P5's planner to the absolute `WISE_ROOT` that
  `make check` then uses.
- The root `requirements.txt` adds the packages P2 (and its pinned harness),
  P5 and the demo asset script import, and the root `.gitignore` covers every problem's run scratch,
  the demo datasets, weights and logs, P5's native builds, raw result JSON
  copied for `make submit` and organizer-only hidden cases.

The fixes that also apply to CROSS's own copy of the problem set were made there
too, on CROSS branch `migrate-cadathlon` after the commit above (now archived
here; see below).

Not migrated: CROSS `main` commit `294ed7d` ("Allow coding agents for all
CADathlon problems"), which moved P1 and P3 into `agent_pset` and is not on
`fix/competition-readiness`, so P1 and P3 stay in `non_agent_pset` (it is archived
here on `archive/cross/main`); the per-run
evidence directories under `validation/` (311 directories, 12,487 files, about
1.8 GB, of which the P1 extraction had already kept
`p1_student_pre_v2/historical_run.json`), which are on the archive branch
`archive/cross/fix/competition-readiness` and linked from `VALIDATION.md` there;
the built `contestant-release.tar.gz` (kept on the archive branches only); CROSS's
root `README.md`; and CROSS's
paper-figure scripts (`figure_drawer/`).

### CROSS history archive (2026-11-14)

CADathlon was then removed from CROSS. Before that, the `cadathlon/` history of
every CROSS branch was copied into this repository, filtered to that directory
with `git filter-repo` (commit messages, authors and dates unchanged). Each
`archive/cross/<branch>` branch reproduces the `cadathlon/` tree of the CROSS
branch it came from byte for byte, including the per-run validation evidence.

| CROSS branch (tip when archived) | Archive branch here (tip) |
|---|---|
| `main` (`294ed7d`) | `archive/cross/main` (`6e75391`) |
| `cadathlon` (`726adf3`) | `archive/cross/cadathlon` (`ed49658`) |
| `fix/cadathlon-validation` (`c835807`) | `archive/cross/fix/cadathlon-validation` (`778077f`) |
| `review/competition-readiness-20261103-090018` (`3e5c01b`) | `archive/cross/review/competition-readiness-20261103-090018` (`7e5ed21`) |
| `feat/p5-wise-tpu-20261105` (`3cec9fa`) | `archive/cross/feat/p5-wise-tpu-20261105` (`6613bad`) |
| `fix/competition-readiness` (`3a49b20`) | `archive/cross/fix/competition-readiness` (`f818ce6`) |
| `migrate-cadathlon` (`d84270c`) | `archive/cross/migrate-cadathlon` (`f9f4ac7`) |

<details><summary>Every archived commit (CROSS commit → archive commit)</summary>

| CROSS | Archive | Committed | Subject |
|---|---|---|---|
| `726adf3` | `ed49658` | 2026-09-24 | Add CADathlon problem sets and benchmark materials |
| `12d72c3` | `5ae2aae` | 2026-10-31 | Share baseline keys and enforce actual-ring security in contest runners |
| `e7a9aa1` | `a26e6fe` | 2026-10-31 | Reduce redundant baseline warmup and model encoding setup |
| `b1c9972` | `8e11c2f` | 2026-10-31 | Evict the baseline client codec on service release |
| `c835807` | `778077f` | 2026-11-01 | Record verified repairs, runtime contracts, and outstanding requirements |
| `c7ec41f` | `b310f53` | 2026-11-03 | Add encrypted transformer references and guarded execution helpers |
| `cde5312` | `bc1fc44` | 2026-11-03 | Add P1 student evaluation and strict calibration scoring |
| `dd9fd70` | `7796adc` | 2026-11-03 | Add P3 compact evaluation and lifecycle cleanup |
| `806f0c9` | `098e909` | 2026-11-03 | Archive runtime regression and secure component evidence |
| `bddedb1` | `a8221a3` | 2026-11-03 | Archive P1 and P3 evaluations and calibration evidence |
| `6d32561` | `6511bf3` | 2026-11-03 | Archive transformer calibration and encrypted component evidence |
| `3e5c01b` | `7e5ed21` | 2026-11-03 | Update release packaging and document remaining readiness gaps |
| `294ed7d` | `6e75391` | 2026-11-04 | Allow coding agents for all CADathlon problems |
| `3cec9fa` | `6613bad` | 2026-11-04 | Add WISE scheduling problem and portable CPU/TPU evaluation |
| `3a49b20` | `f818ce6` | 2026-11-06 | Improve competition runners, compact benchmarks, and validation evidence |
| `152fb80` | `461bb8e` | 2026-11-12 | Add the WISE scheduling problem (P5) to the competition-readiness tree |
| `e52c073` | `33084ab` | 2026-11-12 | Repair P5 packaging, tests and documents found during the migration |
| `d84270c` | `f9f4ac7` | 2026-11-12 | Point the CADathlon README at its new home in cadathlon-p1 |

CROSS commits that changed nothing under `cadathlon/` have no archive commit.
Neither has merge commit `98ca683` (pull request #6), which brought `cadathlon/`
into `main`: after filtering it is identical to `c835807` (`778077f`).
`bce58a7`, the review branch's `jaxite_word` commit, stays reachable in CROSS
only through closed pull request #7.

</details>

In CROSS, `main` removed `cadathlon/` in commit `10ee58b` and its README now
points here. `fix/competition-readiness` removed it in commit `79584c3` and
keeps only the CROSS library changes made during this work: the 21 `jaxite_word`
files of the runtime bundled at this repository's root, for review against
`main` (CROSS `main` itself lacks them, for example
`jaxite_word.mapping.defer_cache_clearing`, which P3 uses). The other CADathlon
branches (`cadathlon`, `fix/cadathlon-validation`,
`review/competition-readiness-20261103-090018`, `feat/p5-wise-tpu-20261105` and
`migrate-cadathlon`) were deleted from CROSS, and its pull requests
[#7](https://github.com/EfficientPPML/CROSS/pull/7) and
[#8](https://github.com/EfficientPPML/CROSS/pull/8) were closed unmerged.
CROSS's history was not rewritten, so older CADathlon commits stay reachable
there: `726adf3` to `294ed7d` in the history of `main` (including the
[#6](https://github.com/EfficientPPML/CROSS/pull/6) merge), `3a49b20` with the
1.8 GB of evidence as the parent of `fix/competition-readiness`, and the review
branch and `3cec9fa` as the heads of the closed pull requests #7 and #8. The
commits of CROSS `migrate-cadathlon` (`152fb80`, `e52c073`, `d84270c`) exist
only in this archive.

CROSS's root `.gitattributes` at `3a49b20` exempted many evidence files from
git's whitespace checks. That file is not part of the `cadathlon/`-only
archive, so `git diff --check` over the archive branches reports whitespace in
those evidence files; their bytes are unchanged.

A plain `git clone` of this repository also downloads the archive branches
(about 0.8 GB); `git clone --single-branch` fetches only the default branch.
