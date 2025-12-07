# run_models_explainers.py
import subprocess
from itertools import product

model_types = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "SVM"]  
explainer_types = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner_confidence", "SQAPlanner_coverage", "SQAPlanner_lift"]

for model, explainer in product(model_types, explainer_types):
    print(f"\n{'='*50}")
    print(f"Running {explainer} on {model}")
    print(f"{'='*50}\n")
    
    subprocess.run([
        "python", "flip_exp.py",
        "--model_type", model,
        "--explainer_type", explainer,
        "--project", "all",
        "--new"
    ])