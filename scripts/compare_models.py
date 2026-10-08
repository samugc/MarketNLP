import json

from marketnlp.data import ROOT, load_prepared
from marketnlp.metrics import compare


def main():
    _, _, manifest = load_prepared()
    results = [json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))
               for name in ("baseline", "finbert")]
    comparison = compare(results, manifest)
    print(f"{'Model':<12} {'Accuracy':>10} {'Macro F1':>10}")
    for name, metrics in comparison["models"].items():
        print(f"{name:<12} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}")
    (ROOT / "results/model_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
