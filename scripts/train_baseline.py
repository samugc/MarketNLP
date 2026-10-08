from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from marketnlp.data import load_prepared
from marketnlp.metrics import save_result


def main():
    train, test, manifest = load_prepared()
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42),
    )
    model.fit([row["text"] for row in train], [row["label"] for row in train])
    predictions = model.predict([row["text"] for row in test]).tolist()
    save_result("baseline", [row["label"] for row in test], predictions, manifest, {
        "vectorizer": {"ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True},
        "classifier": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "random_state": 42},
    })


if __name__ == "__main__":
    main()
