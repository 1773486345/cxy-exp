# PatternAD factorial experiment tools

These scripts turn the six-cell experiment design into reproducible benchmark
commands and summarize only threshold-independent score metrics.

## Execution environment

Every command in this guide uses the canonical global Conda environment:

```text
Conda: /media/h3c/users/shared_app/miniconda3/bin/conda
Name:  patternad_env
Path:  /media/h3c/users/wangyueyang1/.env/envs/patternad_env
Python: 3.8.20
```

Do not search for or create another PatternAD environment unless this documented
prefix is missing. The commands use the shorter `conda run -n patternad_env`
form because the global Conda executable is already on `PATH`.

## Configuration

- `config/patternad/factorial_ablation.json` is the only variant source. It
  defines `A00/A10/A01/A11/B00/B11`, shared training settings, score metrics,
  seeds, and paired comparisons.
- All A cells use complementary conditional scoring with
  `score_mask_ratio=1/3` (three passes). D1 trains a heteroscedastic Gaussian
  with masked NLL and scores conditional two-sided tail surprisal; Student-t is
  intentionally deferred.
- `config/patternad/dataset_groups.json` defines `smoke`, `motivation`,
  `robustness`, and locked `confirmation` groups. SMD entities share one family,
  so they do not receive extra weight in the overall family macro.
- The runnable benchmark smoke group contains Weather and MetroPT3. P1 synthetic
  experiments use the dedicated generator/evaluator below; benchmark
  registration is optional and explicit.

The confirmation group cannot be selected accidentally. It requires all of
`--allow-locked`, explicit `--variant`, explicit `--seeds`, and explicit
`--run-name`. It also requires the complete predeclared dataset/variant/seed
grid and a clean `PatternAD-main` worktree.

## Dry run and execution

Run from the repository root:

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/run_factorial_ablation.py \
  --group smoke --dataset Weather --variant A00 A10 A01 A11 \
  --seeds 2021 --gpus 0 --run-name p0_weather --dry-run
```

Remove `--dry-run` to execute. Values passed with `--gpus` remain physical GPU
IDs; the benchmark backend sets `CUDA_VISIBLE_DEVICES` exactly once.

Each identity is stored below:

```text
result/patternad_strict/<run-name>/<group>/<dataset>/<variant>/seed_<seed>/attempt_NNN/
```

Each completed attempt contains the detailed CSV archive, `benchmark.log`, and
`run_metadata.json`. PatternAD diagnostics are written into the detailed result
first, validated by the runner, and then copied into metadata. A completed cell
must contain ordered calibration/test score calls, finite epoch losses and
scores, best epoch, fit/scoring runtime, and scale boundary statistics. Gaussian
cells fail before `status=completed` when a calibration scale-boundary fraction
reaches the frozen 1% limit. Test boundary fractions remain report-only so they
cannot become a hidden tuning signal. Resume does not accept older completed
attempts without validated diagnostics.

An existing identity is never appended to. Use `--resume` with the same
`--run-name` to skip completed identities and create a new attempt directory for
failed or interrupted identities:

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/run_factorial_ablation.py \
  --group motivation --run-name p2_motivation --gpus 0 --resume
```

Every non-dry run writes `run_plan.json` before starting. It records SHA-256
hashes for data/text files, benchmark config, critical model/evaluation sources,
the manifest, and the expected identity grid. Resume refuses completed results
whose frozen config differs.

Recommended sequence:

1. P0: Weather, one seed, all four A cells. Confirm completion and summarize.
2. P1: run the synthetic generator/evaluator on the frozen generator/model seed grid.
3. P2: `--group motivation`, four A cells, seeds 2021/2022/2023 (the defaults).
4. P3: `--group robustness` for A00/A11, then explicitly run B00/B11 where the
   mask diagnostic is needed.
5. P4: freeze code/config/data hashes first, then run `confirmation` once with
   the locked acknowledgement and the predeclared baseline/candidate only.

Example locked invocation after freezing the candidate:

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/run_factorial_ablation.py \
  --group confirmation --allow-locked --variant A00 A11 \
  --seeds 2021 2022 2023 2024 2025 --gpus 0 \
  --run-name locked_YYYYMMDD
```

## Strict summary

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/summarize_factorial.py \
  --input result/patternad_strict/p2_motivation
```

The summarizer reads only detailed `*.csv.tar.gz` artifacts and writes:

- `entity_seed_score_metrics.csv`
- `entity_seed_run_diagnostics.csv` with epoch/runtime/score/scale diagnostics
- `family_seed_macro.csv` and `family_macro.csv`
- `overall_family_seed_macro.csv`
- entity/seed, family/seed, family-macro, and overall paired delta CSVs

Only `auc_pr`, `VUS_PR`, `auc_roc`, `VUS_ROC`, `R_AUC_PR`, and `R_AUC_ROC`
are accepted. Historical detailed files can repeat these values for several
anomaly-ratio rows; the script verifies that every repeated value is identical
and keeps the first. It never selects a maximum. If a score metric varies across
threshold rows, summarization stops with an error.

The summary must match the exact frozen manifest and complete `run_plan.json`
grid. Failed, missing, unexpected, or config-mismatched cells stop the summary;
they are never silently dropped.

## Hierarchical bootstrap

Run the deterministic family -> entity -> seed bootstrap on either long table:

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/bootstrap_factorial.py \
  --input result/patternad_strict/p2_motivation/summary/paired_entity_seed_delta.csv \
  --n-bootstrap 10000 --seed 2021
```

For `entity_seed_score_metrics.csv`, comparison definitions default to the
frozen factorial manifest. A smaller predeclared set can be supplied before
looking at results by repeating, for example,
`--comparison full_vs_baseline:A11:A00`. Incomplete lhs/rhs pairs fail by
default. `--missing-policy drop` is available only as an explicit sensitivity
analysis and records every dropped identity in `input_diagnostics.csv`.

The bootstrap directory contains:

- `hierarchical_bootstrap.csv`: one row per run/group/comparison/metric with the
  equal-family/equal-entity/equal-seed `mean`, descriptive `paired_row_std`,
  `bootstrap_std`, and percentile 95% CI;
- `input_diagnostics.csv`: unavailable comparisons, pair counts, and any
  explicit drops;
- `bootstrap_metadata.json`: input hash, seed, metric/comparison source, and the
  exact statistic;
- optional `gate_diagnostics.csv`: individual plan criteria, never an aggregate
  stop/go decision.

Small or unbalanced samples are still summarized but carry
`reliability=limited` and machine-readable warnings. The random stream is
derived from the requested seed plus run/group/comparison/metric, so the same
metric is bitwise reproducible even when other requested metrics change.

P2/P4 gate diagnostics are optional and fail closed. Formal gates only accept
`entity_seed_score_metrics.csv`, including its non-empty `plan_hash` and
`config_hash` provenance; a paired CSV cannot reveal an identity that was
uniformly omitted and is therefore restricted to ordinary CI summaries. A
criterion can receive `pass` only when all of the following hold:

- `--missing-policy error` (an explicit drop run can never pass a gate);
- a balanced seed grid and `reliability=standard`;
- `--n-bootstrap` is at least `--minimum-gate-bootstrap` (default 10,000);
- P2 uses resampled families with at least four families and three paired seeds;
- P4 uses fixed family strata with at least two families and five paired seeds.

Otherwise the criterion is `insufficient_data`, even when its observed number
crosses the nominal threshold. A P2 pair diagnostic requires a comparator fixed
by the caller before results are opened:

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/bootstrap_factorial.py \
  --input result/patternad_strict/p2_motivation/summary/entity_seed_score_metrics.csv \
  --comparison full_vs_baseline:A11:A00 \
  --n-bootstrap 10000 --family-mode resample \
  --gate-profile p2 --primary-comparison full_vs_baseline
```

This P2 profile checks only that predeclared pair; it does not call A11's
observed-best competitor the "strongest alternative." Assess the frozen
`A11-A00`, `A11-A10`, and `A11-A01` comparisons separately if the full P2
claim is required. The script never chooses one based on observed values.

P4 has only two predeclared families, so resampling the family level gives a
fragile population-level CI. Keep HAI21 and SMD as fixed strata and bootstrap
entities and seeds within each family instead:

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/summarize_factorial.py \
  --input result/patternad_strict/locked_YYYYMMDD/confirmation

conda run --no-capture-output -n patternad_env \
  python scripts/patternad/bootstrap_factorial.py \
  --input result/patternad_strict/locked_YYYYMMDD/confirmation/summary/entity_seed_score_metrics.csv \
  --comparison full_vs_baseline:A11:A00 \
  --n-bootstrap 10000 --family-mode fixed \
  --gate-profile p4 --primary-comparison full_vs_baseline
```

That P4 CI is conditional on the exact HAI21/SMD strata and must not be
presented as uncertainty over a broader population of dataset families. The
default required families can be replaced by repeated `--required-family`.
Criteria whose required input is absent, such as fixed-threshold normal FPR in
a score-only CSV, are marked `not_evaluated` instead of passing. Freeze every
comparison and gate option before opening confirmation results; selecting them
from observed deltas would itself be test-oracle selection.

The runner fixes the label evaluation to `train_calibration` with a predeclared
1% calibration threshold. Threshold-dependent F1/precision/recall are not part
of this summarizer. The current tests cover strict fit/calibration separation,
static-text handling, target blindness, complete mask coverage, seeded
initialization/mask streams, calibration-only thresholds, and runner contracts.
A tiny GPU smoke produced bitwise-identical scores for the same seed, changed
scores for a different seed, and zero lower/upper scale-boundary hits. The first
real Weather four-cell P0 and its diagnostic A01/A11 rerun completed, but both
predate the visible-context scale-prior prototype and are now pipeline evidence
only. During model development, use one generator seed and an explicit low
`--num-epochs` value before requesting a full experiment.

## P1 contextual synthetic suite

`config/patternad/synthetic_suite.json` freezes a stable two-state switching
VAR and four test mechanisms: same deviation in quiet/volatile context, gradual
drift versus abrupt shift, dependency break with exact per-channel event
marginals, and an unseen stable regime. The official train split is clean and
contains both normal states. Generate the default deterministic replicate with:

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/generate_contextual_synthetic.py
```

An explicit seed writes to an independent `seed_<seed>` directory unless
`--output-dir` is supplied. Each directory contains the resolved config,
generator/evaluator SHA-256 provenance, compressed arrays, event sidecars, and
benchmark-format long CSVs. Seeds 3101-3110 are development; 3111-3120 are
untouched synthetic confirmation and must not be opened during tuning. The full
predeclared generation loop is:

```bash
for seed in 3101 3102 3103 3104 3105 3106 3107 3108 3109 3110 \
            3111 3112 3113 3114 3115 3116 3117 3118 3119 3120; do
  conda run --no-capture-output -n patternad_env \
    python scripts/patternad/generate_contextual_synthetic.py --seed "$seed"
done
```

To install one replicate into the current benchmark, add
`--register-benchmark`. This writes seed-qualified data/text filenames and
idempotently updates `dataset/anomaly_detect/DETECT_META.csv`; omit the flag
when only the direct P1 evaluator is needed.

The evaluator can fit one frozen factorial cell once on the common clean train
split, score all four mechanisms, and then report per-mechanism AP, AP minus
test prevalence, matched ordering, and the quiet/volatile normal FPR gap:

```bash
conda run --no-capture-output -n patternad_env \
  python scripts/patternad/evaluate_contextual_mechanisms.py \
  --artifact-dir artifacts/patternad_synthetic/contextual_v1/seed_3101 \
  --patternad-variant A11 --seed 2021 \
  --output-dir result/patternad_synthetic/A11/generator_seed_3101/model_seed_2021
```

For a quick wiring check only, add `--num-epochs 1`; omit it in reported runs
so the frozen factorial epoch count is used. Run A00/A10/A01/A11 with the same
generator/model seed grid. The evaluator's empirical/conformal-style threshold
uses only the held-out tail of the official clean train split and records
`test_scores_used_for_threshold=false`; test scores never choose a threshold.
Normal-regime FPR pools exclude every injected/reference interval plus a frozen
`seq_len - 1` guard on both sides, so overlapping-window scores influenced by
an event are not counted as ordinary normal false alarms.

The ordering and FPR checks validate only those two contracts; they are not an
overall suite pass. Predeclared AP-minus-prevalence gates cover same-deviation,
drift/shift, and dependency-break, and every mechanism AP remains reportable.
The compared same-deviation and gradual/abrupt windows also have equal injected
squared-deviation magnitude; `raw_magnitude_negative_control` must remain tied.
`context_ood` is deliberately a negative control for a conditional-only score:
success there requires the
separate rare-context score proposed in the research plan. The embedded oracle
is useful for checking generation semantics, not for claiming learned-model
performance.

P1-v1 completed all 120 cells. A11 improved macro AP by `0.063709` over A00
and improved A11-A01 matched ordering by `0.186667`, but reduced the maximum
regime-FPR gap by only `11.46%` against the registered `25%` target. The
transition branch also reduced abrupt/gradual ordering, so its auxiliary loss
is now disabled in all formal cells. Expanded raw cells are stored in
`p1_raw_cells_20260712.tar.gz`; `run_plan.json`, frozen inputs, and the strict
summary remain directly readable.

P1-v2-holdout completed its ten-generator × three-model-seed × four-A-cell
development grid. Its independent 80/10/10 temporal model-fit partition did not
meet the matched-ordering or regime-FPR gates, so do not start P2. The next
development-only check is a causal innovation head: it predicts `x_t` from
`x_{<t}` only and is exported as a diagnostic component, not merged into the
primary score. Run one full-epoch A11 diagnostic first:

```bash
conda run --no-capture-output -n patternad_env \
  python -u scripts/patternad/evaluate_contextual_mechanisms.py \
  --artifact-dir artifacts/patternad_synthetic/contextual_v1/seed_3101 \
  --patternad-variant A11 --seed 2021 --num-epochs 30 \
  --hyperparameter-overrides '{"reconstruction_causal_delta_innovation_loss_weight": 1.0}' \
  --method-name A11_causal_delta_innovation_dev \
  --output-dir result/patternad_synthetic/dev_causal_delta_innovation
```

The preceding level-innovation run is complete and did not pass: its primary
score stayed unchanged by design, while its standardized component got only
`1/2` abrupt/gradual pairs and `0/3` same-deviation pairs right; its learned
scale was approximately constant at `1.0`. The new run instead predicts
`x_t - x_{t-1}` from `x_{<t}` and uses a rolling RMS of prior differences only.

Inspect `score_component_orderings.csv` for
`causal_delta_innovation_standardized_squared_residual`. It must get both
abrupt/gradual pairs right before a multi-seed expansion; also inspect
`predicted_causal_delta_innovation_scale` for causal context sensitivity. Do
not claim a combined detector. If it passes, freeze a multi-seed diagnostic grid
and a p-value combination rule before any real-data run. Do not open
generator seeds 3111-3120 before the development candidate, score combination,
and gates are frozen. Locked synthetic confirmation requires the runner's explicit
`--allow-locked` acknowledgement and the complete predeclared seed grid.

External methods can instead provide one `<mechanism>.npz` per mechanism, each
with a finite `score` vector aligned to the full train-then-test series, and use
`--score-dir`. Omitting both `--score-dir` and `--patternad-variant` evaluates
the embedded generative `oracle_context_score`; that path validates the suite
contract and must not be reported as a learned-model result.
