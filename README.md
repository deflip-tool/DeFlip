# DeFlip-SDP Replication Package

This repository contains the experiment code for the DeFlip study across two settings:

- **JIT-SDP** (`JIT-SDP/`): Just-in-time defect prediction.
- **SDP** (`SDP/`): Software defect prediction.

## Repository Layout

- `JIT-SDP/` – Training scripts, explainers, counterfactual generators, and utilities for
  the JIT study. Dataset and output locations are configured in `JIT-SDP/hyparams.py`.
- `SDP/` – Training, preprocessing, and explainer code for the SDP study. Paths live in
  `SDP/hyparams.py`.
- `plot_rq1.py`, `plot_rq2.py`, `plot_rq3.py` – Plotting utilities used by the paper.
- `replication_cli.py` – An orchestration CLI that wraps the key pipelines for both
  studies.

## Environment Setup

1. Create and activate a Python 3.10+ environment.
2. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (SDP preprocessing only) Install the R package [`Rnalytica`](https://cran.r-project.org/package=Rnalytica)
   and ensure `rpy2` can locate your R installation.

## Datasets

Dataset roots are configured in each study's `hyparams.py` file. Update the paths there to
point to your local copies before running the pipelines. Common locations include:

- **JIT-SDP:**
  - Raw CSV: `JIT-SDP/hyparams.py::JIT_DATASET_PATH`
  - Preprocessed per-release data: `JIT-SDP/hyparams.py::RELEASE_DATASET`
- **SDP:**
  - Original per-project CSVs: `SDP/hyparams.py::ORIGINAL_DATASET`
  - Preprocessed release splits: `SDP/hyparams.py::RELEASE_DATASET` (defaults to
    `./SDP/Dataset/release_dataset` in the current repo state)

## Study script reference

The tables below recap the key scripts and their roles in each study. Use them directly
or through the orchestration CLI shown later.

### JIT-SDP

1. `preprocess_jit.py` – Prepare Apache JIT data.
2. `train_models_jit.py` – Train or evaluate JIT classifiers.
3. `run_explainer.py` – Run LIME/LIME-HPO for explanations.
   - `run_cfexp.py` – CfExplainer neighbourhood + rule mining pipeline.
   - `run_pyexp.py` – PyExplainer neighbourhood search.
4. `plan_explanations.py` – Convert explainer outputs into actionable plans.
   - `plan_final.py` – Build plans for CfExplainer/PyExplainer outputs.
5. `plan_closest.py` – Derive minimum-change plans.
6. `flip_exp.py` – Apply plans to test true positives and measure flips.
7. `flip_closest.py` – Flip evaluation restricted to closest/minimum plans.
8. `cf.py` - Runs modified NICE
   and the DeFlip algorithm to generate counterfactuals.
9. `evaluate.py` / `evaluate_closest.py` – RQ1–RQ3 + implications aggregation for default and
   closest-plan pipelines.

### SDP

1. `preprocess.py` – Preprocess with AutoSpearman feature selection.
2. `train_models.py` – Train RandomForest/XGBoost/SVM models.
3. `run_explainer.py` – Run LIME/LIME-HPO/TimeLIME/SQAPlanner explainers.
4. `mining_sqa_rules.py` – BigML rule mining step required before SQAPlanner plans
   (uses `.env` with `BIGML_USERNAME` and `BIGML_API_KEY`).
5. `generate_closest_plans.py` – Convert explainer outputs into plans (stored under
   `plans_closest/`) and compute importance ratios when requested.
6. `flip_exp.py` – Flip simulation over generated plans.
7. `flip_closest.py` – Flip simulation restricted to closest plans.
8. `cf.py` – Runs modified NICE and the DeFlip
   algorithm to generate counterfactuals.
9. `evaluate.py` / `evaluate_closest.py` – RQ1–RQ3 + implications
   aggregation for full and closest-plan pipelines.

## Command-Line Orchestration

`replication_cli.py` exposes consistent entry points for both studies. Run `--help` on any
command to see available options.

### JIT-SDP

- Train all models:

  ```bash
  python replication_cli.py jit train-models
  ```

- Evaluate previously trained models:

  ```bash
  python replication_cli.py jit train-models --evaluate-only
  ```

- Run LIME/LIME-HPO/CfExplainer/PyExplainer explanations for a specific or all projects:

  ```bash
  python replication_cli.py jit explain \
      --model-type RandomForest \
      --explainer-type CfExplainer \
      --project all
  ```

- Convert explanation outputs into actionable plans. Use `--closest` to build the
  minimum-change variants or omit it for the full plan set. CfExplainer/PyExplainer
  plans are constructed from their mined rules via `plan_final.py`:

  ```bash
  python replication_cli.py jit plan-actions \
      --model-type RandomForest \
      --explainer-type PyExplainer \
      --project all \
      --closest
  ```

- Apply plans to flip predictions (run this before plotting):

  ```bash
  python replication_cli.py jit flip \
      --model-type RandomForest \
      --explainer-type LIME-HPO \
      --project all \
  ```

- Summarize flip experiments for RQ tables/plots (defaults to running all RQs; add
  `--closest` to use the closest-plan pipeline and `--use-default-groups` for RQ3):

  ```bash
  python replication_cli.py jit evaluate --explainer all --models all --projects all
  ```

- Run the full JIT pipeline (train → explain → plan → flip → evaluate) across all
  model/explainer combinations with a single command. Add `--closest` to use the
  closest-plan path and `--evaluate-only` to reuse existing models:

  ```bash
  python replication_cli.py jit run-all --projects all --models all --explainers all
  ```

### SDP

For the SDP study, some explainers require extra steps (e.g., rule mining for
SQAPlanner) and DeFlip counterfactuals are generated via `niceml.py`. The CLI
captures each stage explicitly:

- Preprocess the raw dataset into per-release splits (requires R):

  ```bash
  python replication_cli.py sdp preprocess
  ```

- Train all models (or re-evaluate saved models with `--evaluate-only`). Add
  `--include-extended-models` to also fit the LightGBM and CatBoost models from
  `train_models_new.py` (needed if you plan to generate DeFlip/NICE
  counterfactuals for those learners):

  ```bash
  python replication_cli.py sdp train-models --include-extended-models
  ```

- Run LIME/LIME-HPO/TimeLIME/SQAPlanner explanations:

  ```bash
  python replication_cli.py sdp explain \
      --model-type RandomForest \
      --explainer-type LIME-HPO \
      --project all
  ```

- (SQAPlanner only) Mine association rules on the generated instances before
  creating plans. BigML credentials must be present in `.env`:

  ```bash
  python replication_cli.py sdp mine-rules \
      --model-type SQAPlanner \
      --search-strategy confidence \
      --project activemq@2
  ```

- Convert explanation outputs into closest plans (works for LIME, LIME-HPO,
  TimeLIME, and SQAPlanner; include `--search-strategy` if your explanations
  were stored under a subfolder such as `confidence/`):

  ```bash
  python replication_cli.py sdp plan-actions \
      --model-type RandomForest \
      --explainer-type LIME-HPO \
      --project all \
  ```

  Use `--compute-importance` to compute feature-importance ratios instead of
  writing plan files.

- Generate DeFlip/NICE counterfactuals (writes `experiments/{project}/{model}/CF_all.csv`):

  ```bash
  python replication_cli.py sdp counterfactuals \
      --project all \
      --model-types RandomForest,SVM,XGBoost \
      --max-features 5 \
      --distance unit_l2
  ```

- Apply plans to flip predictions (run this before plotting the plan-based
  explainers). Add `--closest` to read from `plans_closest/` and
  `experiments_closest/`:

  ```bash
  python replication_cli.py sdp flip \
      --model-type RandomForest \
      --explainer-type LIME-HPO \
      --project all \
  ```

- Summarize flip experiments and DeFlip counterfactuals for plotting:

  ```bash
  python replication_cli.py sdp evaluate --explainer all --models all --projects all
  ```

- Run the full SDP pipeline (preprocess → train → explain → plan → flip → optional
  counterfactuals → evaluate) across all model/explainer combinations. Add
  `--closest` to restrict to closest plans, `--include-counterfactuals` to also run
  `niceml.py`, and `--include-extended-models` to build LightGBM/CatBoost models:

  ```bash
  python replication_cli.py sdp run-all --projects all --models all --explainers all \
      --include-extended-models --include-counterfactuals
  ```

## Plotting

Once experiments finish, the `plot_rq*.py` scripts can be used to regenerate the figures
from the paper. Each script reads the CSV outputs written by the training/explanation
pipelines (see the constants in the script bodies for expected locations).

## Notes and Tips

- Paths in the study-specific `hyparams.py` files are relative to the repository root by
  default. Adjust them as needed for your environment.
- The explainers use multiprocessing; ensure your machine has enough memory to spawn
  concurrent workers when running the explanation commands.
- For quick smoke tests, you can point the dataset paths to a reduced sample and run the
  CLI with the same commands shown above.
- The overall workflow for actionability is: (1) run explainers, (2) generate plans using
  the new `plan-actions` commands, and (3) flip simulation (4) evaluate using the study-specific scripts
  such as `SDP/evaluate_cf.py` or the `evaluate_*` utilities under `JIT-SDP/`. Some
  explainers require extra manual steps (e.g., SQAPlanner needs BigML credentials for
  rule mining via `SDP/mining_sqa_rules.py`). Refer to the inline comments inside each
  script for the exact sequencing.
