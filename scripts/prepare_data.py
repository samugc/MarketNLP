import json
from collections import Counter

from huggingface_hub import hf_hub_download

from marketnlp.data import DATA, DATASET, REVISION, file_hash, prepare_splits, read_csv, write_csv


def main():
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = [hf_hub_download(DATASET, filename, repo_type="dataset", revision=REVISION,
                             local_dir=raw)
             for filename in ("sent_train.csv", "sent_valid.csv")]
    original = [read_csv(path, source_labels=True) for path in paths]
    train, test = prepare_splits(*original)
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": "published train; published validation used as final test, no tuning",
        "cleaning": "remove URLs, collapse whitespace, case-insensitive deduplication; remove within-split "
                    "label conflicts and test texts seen in cleaned train",
        "source_rows": dict(zip(("train", "test"), map(len, original))),
        "rows": {}, "class_counts": {}, "sha256": {},
    }
    for name, rows in (("train", train), ("test", test)):
        path = DATA / f"{name}.csv"
        write_csv(path, rows)
        manifest["rows"][name] = len(rows)
        manifest["class_counts"][name] = dict(sorted(Counter(row["label"] for row in rows).items()))
        manifest["sha256"][name] = file_hash(path)
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
