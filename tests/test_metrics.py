import pytest

from marketnlp.metrics import evaluate


def test_metrics_include_all_classes_and_confusion_order():
    scores = evaluate(["negative", "neutral", "positive", "positive"],
                      ["negative", "neutral", "neutral", "positive"])
    assert scores["accuracy"] == 0.75
    assert scores["macro_f1"] == pytest.approx(7 / 9)
    assert scores["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert scores["per_class"]["positive"]["recall"] == 0.5
    assert evaluate(["neutral"], ["neutral"])["macro_f1"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("truth, predictions", [([], []), (["neutral"], []),
                                               (["positive"], ["LABEL_0"])])
def test_invalid_metrics_inputs(truth, predictions):
    with pytest.raises(ValueError):
        evaluate(truth, predictions)


def test_comparison_rejects_stale_data():
    from marketnlp.metrics import compare

    manifest = {"sha256": {"test": "abc"}, "rows": {"test": 2}}
    results = [{"model": model, "data": manifest,
                "metrics": {"accuracy": 0.5, "macro_f1": 0.4, "n_test": 2}}
               for model in ("baseline", "finbert")]
    assert compare(results, manifest)["models"]["baseline"]["accuracy"] == 0.5
    results[1]["data"] = {"sha256": {"test": "changed"}, "rows": {"test": 2}}
    with pytest.raises(ValueError, match="stale"):
        compare(results, manifest)
