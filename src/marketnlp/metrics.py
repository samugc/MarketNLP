import json
import platform
from importlib.metadata import version

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from marketnlp.data import LABELS, ROOT


def evaluate(y_true, y_pred):
    if not len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Expected equally sized, nonempty labels and predictions")
    if (set(y_true) | set(y_pred)) - set(LABELS):
        raise ValueError("Unknown sentiment label")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "per_class": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "label_order": LABELS,
        "n_test": len(y_true),
    }


def save_result(name, y_true, y_pred, manifest, settings):
    result = {
        "model": name,
        "data": manifest,
        "settings": settings,
        "metrics": evaluate(y_true, y_pred),
        "environment": {
            "python": platform.python_version(),
            **{package: version(package) for package in
               ("scikit-learn", "torch", "transformers", "huggingface-hub")},
        },
    }
    directory = ROOT / "results"
    directory.mkdir(exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{name}: accuracy={result['metrics']['accuracy']:.4f} "
          f"macro_f1={result['metrics']['macro_f1']:.4f}")
    return result


def compare(results, manifest):
    if {result["model"] for result in results} != {"baseline", "finbert"} or len(results) != 2:
        raise ValueError("Expected one baseline and one FinBERT result")
    for result in results:
        if result["data"] != manifest or result["metrics"]["n_test"] != manifest["rows"]["test"]:
            raise ValueError("Results use different or stale data; rerun both models")
    return {
        "data": manifest,
        "models": {result["model"]: {metric: result["metrics"][metric]
                   for metric in ("accuracy", "macro_f1")} for result in results},
    }
