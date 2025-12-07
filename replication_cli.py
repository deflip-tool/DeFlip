"""Unified command-line entry points for the DeFlip experiments."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Tuple

BASE_DIR = Path(__file__).resolve().parent
JIT_MODELS = ["RandomForest", "SVM", "XGBoost", "LightGBM", "CatBoost"]
JIT_EXPLAINERS = ["LIME", "LIME-HPO", "CfExplainer", "PyExplainer"]
SDP_MODELS = ["RandomForest", "XGBoost", "SVM", "LightGBM", "CatBoost"]
SDP_EXPLAINERS = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"]


@contextmanager
def study_context(study_folder: str):
    """Temporarily add a study folder to ``sys.path`` and yield its Path."""

    study_path = BASE_DIR / study_folder
    sys.path.insert(0, str(study_path))
    try:
        yield study_path
    finally:
        sys.path.remove(str(study_path))


def load_module(module_path: Path, name: str):
    """Load a module from ``module_path`` under a temporary name."""

    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(script_path: Path, args: list[str]):
    """Execute a study script via ``python <script> ...``."""

    subprocess.run([sys.executable, str(script_path), *args], check=True)


def resolve_projects(projects: dict, requested: str) -> Iterable[str]:
    if requested == "all":
        return list(sorted(projects.keys()))
    return [project.strip() for project in requested.split(",") if project.strip()]


def resolve_list(requested: str, defaults: list[str]) -> list[str]:
    """Return a list of values from a comma/space separated string or ``all``."""

    if requested.lower() == "all":
        return defaults
    splitters = [",", " "]
    items: list[str] = [requested]
    for splitter in splitters:
        new_items = []
        for item in items:
            new_items.extend([part for part in item.split(splitter) if part])
        items = new_items
    return [item.strip() for item in items if item.strip()]


# ---------------------------- JIT-SDP tasks ---------------------------------

def jit_train_models(evaluate_only: bool = False):
    with study_context("JIT-SDP") as study_path:
        trainer = load_module(study_path / "train_models_jit.py", "jit_train")
        if evaluate_only:
            trainer.eval_all_project()
        else:
            trainer.train_all_project()


def jit_run_explanations(model_type: str, explainer_type: str, project: str):
    with study_context("JIT-SDP") as study_path:
        # CfExplainer and PyExplainer have dedicated runners; LIME/LIME-HPO reuse run_explainer.py
        if explainer_type == "CfExplainer":
            args = ["--model_type", model_type, "--project", project]
            run_script(study_path / "run_cfexp.py", args)
            return

        if explainer_type == "PyExplainer":
            args = ["--model_type", model_type, "--project", project]
            run_script(study_path / "run_pyexp.py", args)
            return

        data_utils = load_module(study_path / "data_utils.py", "jit_data_utils")
        runner = load_module(study_path / "run_explainer.py", "jit_run_explainer")
        projects = data_utils.read_dataset()
        for project_name in resolve_projects(projects, project):
            train, test = projects[project_name]
            runner.run_single_project(train, test, project_name, model_type, explainer_type)


def jit_plan_actions(
    model_type: str,
    explainer_type: str,
    project: str,
    search_strategy: str | None,
    compute_importance: bool,
    closest: bool,
):
    """Derive actionable plans from previously saved JIT explanations."""

    with study_context("JIT-SDP") as study_path:
        # CfExplainer / PyExplainer plans are built via plan_final.py using saved neighbourhoods/rules.
        if explainer_type in {"CfExplainer", "PyExplainer"}:
            if compute_importance:
                raise ValueError("--compute-importance is not supported for CfExplainer/PyExplainer planning.")
            args = [
                "--model_type",
                model_type,
                "--explainer_name",
                explainer_type,
                "--project",
                project,
            ]
            run_script(study_path / "plan_final.py", args)
            return

        planner_name = "plan_closest.py" if closest else "plan_explanations.py"
        planner = load_module(study_path / planner_name, "jit_plan")
        projects = planner.read_dataset()

        for project_name in resolve_projects(projects, project):
            train, test = projects[project_name]
            if compute_importance:
                planner.get_importance_ratio(train, test, project_name, model_type, explainer_type, verbose=True)
            else:
                planner.run_single(
                    train,
                    test,
                    project_name,
                    model_type,
                    explainer_type,
                    search_strategy,
                    verbose=True,
                )


def jit_flip(
    model_type: str,
    explainer_type: str,
    project: str,
    search_strategy: str | None,
    verbose: bool,
    fresh: bool,
    get_flip_rate: bool,
    closest: bool,
):
    script = "flip_closest.py" if closest else "flip_exp.py"
    with study_context("JIT-SDP") as study_path:
        args = ["--explainer_type", explainer_type, "--model_type", model_type, "--project", project]
        if search_strategy:
            args.extend(["--search_strategy", search_strategy])
        if verbose:
            args.append("--verbose")
        if fresh:
            args.append("--new")
        if get_flip_rate:
            args.append("--get_flip_rate")
        run_script(study_path / script, args)


# ------------------------------ SDP tasks -----------------------------------

def sdp_preprocess():
    with study_context("SDP") as study_path:
        preprocess = load_module(study_path / "preprocess.py", "sdp_preprocess")
        preprocess.organize_original_dataset()
        preprocess.prepare_release_dataset()


def sdp_train_models(evaluate_only: bool = False, include_extended: bool = False):
    with study_context("SDP") as study_path:
        trainer = load_module(study_path / "train_models.py", "sdp_train")
        if evaluate_only:
            trainer.eval_all_project()
        else:
            trainer.train_all_project()

        if include_extended and not evaluate_only:
            extended = load_module(study_path / "train_models_new.py", "sdp_train_extended")
            extended.main()


def sdp_run_explanations(model_type: str, explainer_type: str, project: str):
    with study_context("SDP") as study_path:
        data_utils = load_module(study_path / "data_utils.py", "sdp_data_utils")
        runner = load_module(study_path / "run_explainer.py", "sdp_run_explainer")
        projects = data_utils.read_dataset()
        for project_name in resolve_projects(projects, project):
            train, test = projects[project_name]
            runner.run_single_project(train, test, project_name, model_type, explainer_type)


def sdp_plan_actions(
    model_type: str, explainer_type: str, project: str, search_strategy: str | None, compute_importance: bool
):
    """Derive actionable plans from previously saved SDP explanations."""

    with study_context("SDP") as study_path:
        planner = load_module(study_path / "generate_closest_plans.py", "sdp_plan_closest")
        projects = planner.read_dataset()

        for project_name in resolve_projects(projects, project):
            train, test = projects[project_name]
            if compute_importance:
                planner.get_importance_ratio(train, test, project_name, model_type, explainer_type, verbose=True)
            else:
                planner.run_single(
                    train,
                    test,
                    project_name,
                    model_type,
                    explainer_type,
                    search_strategy,
                    verbose=True,
                )


def sdp_mine_sqa_rules(project: str, search_strategy: str, model_type: str):
    """Trigger BigML rule mining for SQAPlanner before creating plans."""

    with study_context("SDP") as study_path:
        projects = load_module(study_path / "data_utils.py", "sdp_data_utils").read_dataset()
        for project_name in resolve_projects(projects, project):
            run_script(
                study_path / "mining_sqa_rules.py",
                [
                    "--project",
                    project_name,
                    "--search_strategy",
                    search_strategy,
                    "--model_type",
                    model_type,
                ],
            )


def sdp_generate_counterfactuals(
    project: str,
    model_types: str,
    max_features: int,
    exact_k: bool,
    distance: str,
    nice_distance_metric: str,
    overwrite: bool,
    verbose: bool,
):
    """Run DeFlip/NICE counterfactual generation (niceml.py)."""

    with study_context("SDP") as study_path:
        args = [
            "--project",
            project,
            "--model_types",
            model_types,
            "--max_features",
            str(max_features),
            "--distance",
            distance,
            "--nice_distance_metric",
            nice_distance_metric,
        ]
        if exact_k:
            args.append("--exact_k")
        if overwrite:
            args.append("--overwrite")
        if verbose:
            args.append("--verbose")

        run_script(study_path / "niceml.py", args)


def jit_run_all(
    models: str,
    explainers: str,
    projects: str,
    search_strategy: str | None,
    closest: bool,
    evaluate_only: bool,
):
    """Run the full JIT-SDP pipeline across models/explainers/projects."""

    model_list = resolve_list(models, JIT_MODELS)
    explainer_list = resolve_list(explainers, JIT_EXPLAINERS)

    jit_train_models(evaluate_only=evaluate_only)

    for model in model_list:
        for explainer in explainer_list:
            jit_run_explanations(model, explainer, projects)
            jit_plan_actions(model, explainer, projects, search_strategy, False, closest)
            jit_flip(model, explainer, projects, search_strategy, False, False, False, closest)

    jit_evaluate(
        closest,
        rq1=True,
        rq2=True,
        rq3=True,
        implications=True,
        explainer=" ".join(explainer_list),
        distance="mahalanobis",
        models=",".join(model_list),
        projects=projects,
        use_default_groups=True,
    )


def sdp_run_all(
    models: str,
    explainers: str,
    projects: str,
    search_strategy: str | None,
    closest: bool,
    include_extended_models: bool,
    skip_preprocess: bool,
    skip_training: bool,
    include_counterfactuals: bool,
):
    """Run the full SDP pipeline across models/explainers/projects."""

    model_list = resolve_list(models, SDP_MODELS)
    explainer_list = resolve_list(explainers, SDP_EXPLAINERS)

    if not skip_preprocess:
        sdp_preprocess()

    if not skip_training:
        sdp_train_models(evaluate_only=False, include_extended=include_extended_models)

    for model in model_list:
        for explainer in explainer_list:
            sdp_run_explanations(model, explainer, projects)
            if explainer == "SQAPlanner":
                sdp_mine_sqa_rules(projects, search_strategy or "confidence", model)
            sdp_plan_actions(model, explainer, projects, search_strategy, False)
            sdp_flip(model, explainer, projects, search_strategy, False, False, False, closest)

    if include_counterfactuals:
        sdp_generate_counterfactuals(
            project=projects,
            model_types=",".join(model_list),
            max_features=5,
            exact_k=False,
            distance="unit_l2",
            nice_distance_metric="HEOM",
            overwrite=False,
            verbose=False,
        )

    sdp_evaluate(
        rq1=True,
        rq2=True,
        rq3=True,
        implications=True,
        explainer=" ".join(explainer_list),
        distance="mahalanobis",
        models=",".join(model_list),
        projects=projects,
        use_default_groups=True,
    )


def jit_evaluate(
    closest: bool,
    rq1: bool,
    rq2: bool,
    rq3: bool,
    implications: bool,
    explainer: str,
    distance: str,
    models: str,
    projects: str,
    use_default_groups: bool,
):
    script = "evaluate_closest.py" if closest else "evaluate_final.py"
    with study_context("JIT-SDP") as study_path:
        args = []
        # If no specific RQ flags are provided, run all.
        if not any([rq1, rq2, rq3, implications]):
            rq1 = rq2 = rq3 = implications = True

        for flag, enabled in (
            ("--rq1", rq1),
            ("--rq2", rq2),
            ("--rq3", rq3),
            ("--implications", implications),
        ):
            if enabled:
                args.append(flag)

        args.extend([
            "--explainer",
            explainer,
            "--distance",
            distance,
            "--models",
            models,
            "--projects",
            projects,
        ])
        if use_default_groups:
            args.append("--use_default_groups")
        run_script(study_path / script, args)


def sdp_evaluate(
    rq1: bool,
    rq2: bool,
    rq3: bool,
    implications: bool,
    explainer: str,
    distance: str,
    models: str,
    projects: str,
    use_default_groups: bool,
):
    with study_context("SDP") as study_path:
        args = []
        if not any([rq1, rq2, rq3, implications]):
            rq1 = rq2 = rq3 = implications = True

        for flag, enabled in (
            ("--rq1", rq1),
            ("--rq2", rq2),
            ("--rq3", rq3),
            ("--implications", implications),
        ):
            if enabled:
                args.append(flag)

        args.extend([
            "--explainer",
            explainer,
            "--distance",
            distance,
            "--models",
            models,
            "--projects",
            projects,
        ])
        if use_default_groups:
            args.append("--use_default_groups")
        run_script(study_path / "evaluate_cf.py", args)


def sdp_flip(
    model_type: str,
    explainer_type: str,
    project: str,
    search_strategy: str | None,
    verbose: bool,
    fresh: bool,
    get_flip_rate: bool,
    closest: bool,
):
    script = "flip_closest.py" if closest else "flip_exp.py"
    with study_context("SDP") as study_path:
        args = ["--explainer_type", explainer_type, "--model_type", model_type, "--project", project]
        if search_strategy:
            args.extend(["--search_strategy", search_strategy])
        if verbose:
            args.append("--verbose")
        if fresh:
            args.append("--new")
        if get_flip_rate:
            args.append("--get_flip_rate")
        run_script(study_path / script, args)


# ----------------------------- CLI parsing ----------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="study", required=True)

    # JIT-SDP
    jit_parser = subparsers.add_parser("jit", help="JIT-SDP tasks")
    jit_subparsers = jit_parser.add_subparsers(dest="command", required=True)

    jit_train = jit_subparsers.add_parser("train-models", help="Train or evaluate JIT models")
    jit_train.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and only run evaluation on saved models.",
    )

    jit_explain = jit_subparsers.add_parser("explain", help="Run JIT explainers")
    jit_explain.add_argument(
        "--model-type",
        default="RandomForest",
        choices=JIT_MODELS,
        help="Classifier to load for explanation.",
    )
    jit_explain.add_argument(
        "--explainer-type",
        default="LIME-HPO",
        choices=JIT_EXPLAINERS,
        help="Which explainer to run.",
    )
    jit_explain.add_argument(
        "--project",
        default="all",
        help="Project name or comma-separated list; use 'all' for every project.",
    )

    jit_plan = jit_subparsers.add_parser("plan-actions", help="Create actionable plans from explanations")
    jit_plan.add_argument(
        "--model-type",
        default="RandomForest",
        choices=JIT_MODELS,
        help="Classifier used to produce the explanations.",
    )
    jit_plan.add_argument(
        "--explainer-type",
        default="LIME-HPO",
        choices=JIT_EXPLAINERS,
        help="Explainer whose outputs will be converted into plans.",
    )
    jit_plan.add_argument(
        "--project",
        default="all",
        help="Project name or comma-separated list; use 'all' for every project.",
    )
    jit_plan.add_argument(
        "--search-strategy",
        default=None,
        help="Optional search strategy suffix used when running explanations (e.g., 'confidence').",
    )
    jit_plan.add_argument(
        "--compute-importance",
        action="store_true",
        help="Compute feature-importance ratios instead of generating plans.",
    )
    jit_plan.add_argument(
        "--closest",
        action="store_true",
        help="Use the closest-plan generator (plan_closest.py) instead of full plan_explanations.py",
    )

    jit_flip_cmd = jit_subparsers.add_parser("flip", help="Apply plans to flip predictions")
    jit_flip_cmd.add_argument("--model-type", default="RandomForest", choices=JIT_MODELS)
    jit_flip_cmd.add_argument("--explainer-type", default="LIME-HPO", choices=JIT_EXPLAINERS + ["CF"])
    jit_flip_cmd.add_argument("--project", default="all", help="Project name or comma-separated list; use 'all' for every project.")
    jit_flip_cmd.add_argument("--search-strategy", default=None, help="Optional search strategy suffix used when running explanations (e.g., 'confidence').")
    jit_flip_cmd.add_argument("--verbose", action="store_true", help="Show per-instance progress")
    jit_flip_cmd.add_argument("--fresh", action="store_true", help="Ignore cached flip CSVs and recompute")
    jit_flip_cmd.add_argument("--get-flip-rate", action="store_true", help="Only compute aggregate flip rates")
    jit_flip_cmd.add_argument("--closest", action="store_true", help="Use the closest-plan pipeline (plans_closest/experiments_closest)")

    jit_eval_cmd = jit_subparsers.add_parser("evaluate", help="Summarize flip experiments for plotting")
    for flag in ("rq1", "rq2", "rq3", "implications"):
        jit_eval_cmd.add_argument(f"--{flag}", action="store_true", help=f"Run {flag.upper()} aggregation")
    jit_eval_cmd.add_argument("--explainer", default="all", help="Explainers spaced/comma (e.g., 'CF LIME'), or 'all'")
    jit_eval_cmd.add_argument("--distance", default="mahalanobis", choices=["mahalanobis", "cosine"])
    jit_eval_cmd.add_argument("--models", default="RandomForest,SVM,XGBoost,CatBoost,LightGBM", help='Models spaced/comma (e.g., "SVM RF"), or "all"')
    jit_eval_cmd.add_argument("--projects", default="all", help='Projects spaced/comma, or "all"')
    jit_eval_cmd.add_argument("--use-default-groups", action="store_true", help="Use predefined release groups when aggregating RQ3")
    jit_eval_cmd.add_argument("--closest", action="store_true", help="Evaluate closest-plan outputs instead of the default pipeline")

    jit_run_all_cmd = jit_subparsers.add_parser(
        "run-all", help="Run the full JIT pipeline across all model/explainer combinations"
    )
    jit_run_all_cmd.add_argument("--projects", default="all", help="Project name(s) or 'all'.")
    jit_run_all_cmd.add_argument("--models", default="all", help="Comma/space separated list of models or 'all'.")
    jit_run_all_cmd.add_argument(
        "--explainers", default="all", help="Comma/space separated list of explainers or 'all'."
    )
    jit_run_all_cmd.add_argument(
        "--search-strategy",
        default=None,
        help="Optional search strategy suffix used when running explanations (e.g., 'confidence').",
    )
    jit_run_all_cmd.add_argument("--closest", action="store_true", help="Use closest-plan pipeline where applicable.")
    jit_run_all_cmd.add_argument("--evaluate-only", action="store_true", help="Skip JIT training and only evaluate saved models.")

    # SDP
    sdp_parser = subparsers.add_parser("sdp", help="Static defect prediction tasks")
    sdp_subparsers = sdp_parser.add_subparsers(dest="command", required=True)

    sdp_subparsers.add_parser("preprocess", help="Prepare release datasets")

    sdp_train = sdp_subparsers.add_parser("train-models", help="Train or evaluate SDP models")
    sdp_train.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and only run evaluation on saved models.",
    )
    sdp_train.add_argument(
        "--include-extended-models",
        action="store_true",
        help="Also train LightGBM and CatBoost models via train_models_new.py.",
    )

    sdp_explain = sdp_subparsers.add_parser(
        "explain", help="Run SDP explainers (LIME, LIME-HPO, TimeLIME, SQAPlanner)"
    )
    sdp_explain.add_argument(
        "--model-type",
        default="RandomForest",
        choices=SDP_MODELS,
        help="Classifier to load for explanation.",
    )
    sdp_explain.add_argument(
        "--explainer-type",
        default="LIME-HPO",
        choices=SDP_EXPLAINERS,
        help="Which explainer to run.",
    )
    sdp_explain.add_argument(
        "--project",
        default="all",
        help="Project name or comma-separated list; use 'all' for every project.",
    )

    sdp_plan = sdp_subparsers.add_parser(
        "plan-actions", help="Create actionable plans from SDP explanations"
    )
    sdp_plan.add_argument(
        "--model-type",
        default="RandomForest",
        choices=SDP_MODELS,
        help="Classifier used to produce the explanations.",
    )
    sdp_plan.add_argument(
        "--explainer-type",
        default="LIME-HPO",
        choices=SDP_EXPLAINERS,
        help="Explainer whose outputs will be converted into plans.",
    )
    sdp_plan.add_argument(
        "--project",
        default="all",
        help="Project name or comma-separated list; use 'all' for every project.",
    )
    sdp_plan.add_argument(
        "--search-strategy",
        default=None,
        help="Optional search strategy suffix used when running explanations (e.g., 'confidence').",
    )
    sdp_plan.add_argument(
        "--compute-importance",
        action="store_true",
        help="Compute feature-importance ratios instead of generating plans.",
    )

    sdp_mine = sdp_subparsers.add_parser(
        "mine-rules",
        help="Run SQAPlanner rule mining on generated instances (requires BigML credentials)",
    )
    sdp_mine.add_argument(
        "--project", default="all", help="Project name or comma-separated list; use 'all' for every project."
    )
    sdp_mine.add_argument(
        "--search-strategy",
        default="confidence",
        choices=["coverage", "confidence", "lift"],
        help="Association rule search strategy used by SQAPlanner.",
    )
    sdp_mine.add_argument("--model-type", default="RandomForest", help="Model used to generate SQAPlanner neighbors.")

    sdp_cf = sdp_subparsers.add_parser("counterfactuals", help="Run DeFlip/NICE counterfactual generation")
    sdp_cf.add_argument(
        "--project", default="all", help="Project name or comma-separated list; use 'all' for every project."
    )
    sdp_cf.add_argument(
        "--model-types",
        default="RandomForest,SVM,XGBoost,LightGBM,CatBoost",
        help="Comma/space separated list of model types to process.",
    )
    sdp_cf.add_argument("--max-features", type=int, default=5, help="Maximum features edited per CF (K).")
    sdp_cf.add_argument("--exact-k", action="store_true", help="Force exactly K edits instead of ≤K.")
    sdp_cf.add_argument(
        "--distance",
        default="unit_l2",
        choices=["unit_l2", "euclidean", "raw_l2"],
        help="Distance metric reported and used for ranking.",
    )
    sdp_cf.add_argument(
        "--nice-distance-metric",
        default="HEOM",
        help="Distance metric passed to the NICE generator (see niceml.py for options).",
    )
    sdp_cf.add_argument("--overwrite", action="store_true", help="Regenerate CF_all.csv even if it exists.")
    sdp_cf.add_argument("--verbose", action="store_true", help="Show per-project progress.")

    sdp_flip_cmd = sdp_subparsers.add_parser("flip", help="Apply plans to flip predictions")
    sdp_flip_cmd.add_argument("--model-type", default="RandomForest", choices=SDP_MODELS)
    sdp_flip_cmd.add_argument("--explainer-type", default="LIME-HPO", choices=SDP_EXPLAINERS + ["CF"])
    sdp_flip_cmd.add_argument("--project", default="all", help="Project name or comma-separated list; use 'all' for every project.")
    sdp_flip_cmd.add_argument("--search-strategy", default=None, help="Optional search strategy suffix used when running explanations (e.g., 'confidence').")
    sdp_flip_cmd.add_argument("--verbose", action="store_true", help="Show per-instance progress")
    sdp_flip_cmd.add_argument("--fresh", action="store_true", help="Ignore cached flip CSVs and recompute")
    sdp_flip_cmd.add_argument("--get-flip-rate", action="store_true", help="Only compute aggregate flip rates")
    sdp_flip_cmd.add_argument("--closest", action="store_true", help="Use the closest-plan pipeline (plans_closest/experiments_closest)")

    sdp_eval_cmd = sdp_subparsers.add_parser("evaluate", help="Summarize flip experiments for plotting")
    for flag in ("rq1", "rq2", "rq3", "implications"):
        sdp_eval_cmd.add_argument(f"--{flag}", action="store_true", help=f"Run {flag.upper()} aggregation")
    sdp_eval_cmd.add_argument("--explainer", default="all", help="Explainers spaced/comma (e.g., 'CF LIME'), or 'all'")
    sdp_eval_cmd.add_argument("--distance", default="mahalanobis", choices=["mahalanobis", "cosine"])
    sdp_eval_cmd.add_argument("--models", default="RandomForest,SVM,XGBoost,CatBoost,LightGBM", help='Models spaced/comma (e.g., "SVM RF"), or "all"')
    sdp_eval_cmd.add_argument("--projects", default="all", help='Projects spaced/comma, or "all"')
    sdp_eval_cmd.add_argument("--use-default-groups", action="store_true", help="Use predefined release groups when aggregating RQ3")

    sdp_run_all_cmd = sdp_subparsers.add_parser(
        "run-all", help="Run the full SDP pipeline across all model/explainer combinations"
    )
    sdp_run_all_cmd.add_argument("--projects", default="all", help="Project name(s) or 'all'.")
    sdp_run_all_cmd.add_argument("--models", default="all", help="Comma/space separated list of models or 'all'.")
    sdp_run_all_cmd.add_argument(
        "--explainers", default="all", help="Comma/space separated list of explainers or 'all'."
    )
    sdp_run_all_cmd.add_argument(
        "--search-strategy",
        default=None,
        help="Optional search strategy suffix used when running explanations (e.g., 'confidence').",
    )
    sdp_run_all_cmd.add_argument("--closest", action="store_true", help="Use closest-plan pipeline where applicable.")
    sdp_run_all_cmd.add_argument(
        "--include-extended-models",
        action="store_true",
        help="Also train LightGBM and CatBoost models via train_models_new.py during the pipeline.",
    )
    sdp_run_all_cmd.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing stage.")
    sdp_run_all_cmd.add_argument("--skip-training", action="store_true", help="Skip training stage.")
    sdp_run_all_cmd.add_argument(
        "--include-counterfactuals",
        action="store_true",
        help="Run DeFlip/NICE counterfactual generation after plan-based flips.",
    )

    return parser


def main(argv: Tuple[str, ...] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.study == "jit":
        if args.command == "train-models":
            jit_train_models(evaluate_only=args.evaluate_only)
        elif args.command == "explain":
            jit_run_explanations(args.model_type, args.explainer_type, args.project)
        elif args.command == "plan-actions":
            jit_plan_actions(
                args.model_type,
                args.explainer_type,
                args.project,
                args.search_strategy,
                args.compute_importance,
                args.closest,
            )
        elif args.command == "flip":
            jit_flip(
                args.model_type,
                args.explainer_type,
                args.project,
                args.search_strategy,
                args.verbose,
                args.fresh,
                args.get_flip_rate,
                args.closest,
            )
        elif args.command == "evaluate":
            jit_evaluate(
                args.closest,
                args.rq1,
                args.rq2,
                args.rq3,
                args.implications,
                args.explainer,
                args.distance,
                args.models,
                args.projects,
                args.use_default_groups,
            )
        elif args.command == "run-all":
            jit_run_all(
                models=args.models,
                explainers=args.explainers,
                projects=args.projects,
                search_strategy=args.search_strategy,
                closest=args.closest,
                evaluate_only=args.evaluate_only,
            )

    elif args.study == "sdp":
        if args.command == "preprocess":
            sdp_preprocess()
        elif args.command == "train-models":
            sdp_train_models(evaluate_only=args.evaluate_only, include_extended=args.include_extended_models)
        elif args.command == "explain":
            sdp_run_explanations(args.model_type, args.explainer_type, args.project)
        elif args.command == "plan-actions":
            sdp_plan_actions(
                args.model_type, args.explainer_type, args.project, args.search_strategy, args.compute_importance
            )
        elif args.command == "mine-rules":
            sdp_mine_sqa_rules(args.project, args.search_strategy, args.model_type)
        elif args.command == "counterfactuals":
            sdp_generate_counterfactuals(
                args.project,
                args.model_types,
                args.max_features,
                args.exact_k,
                args.distance,
                args.nice_distance_metric,
                args.overwrite,
                args.verbose,
            )
        elif args.command == "flip":
            sdp_flip(
                args.model_type,
                args.explainer_type,
                args.project,
                args.search_strategy,
                args.verbose,
                args.fresh,
                args.get_flip_rate,
                args.closest,
            )
        elif args.command == "evaluate":
            sdp_evaluate(
                args.rq1,
                args.rq2,
                args.rq3,
                args.implications,
                args.explainer,
                args.distance,
                args.models,
                args.projects,
                args.use_default_groups,
            )
        elif args.command == "run-all":
            sdp_run_all(
                models=args.models,
                explainers=args.explainers,
                projects=args.projects,
                search_strategy=args.search_strategy,
                closest=args.closest,
                include_extended_models=args.include_extended_models,
                skip_preprocess=args.skip_preprocess,
                skip_training=args.skip_training,
                include_counterfactuals=args.include_counterfactuals,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
