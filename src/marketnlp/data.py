import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATASET = "zeroshot/twitter-financial-news-sentiment"
REVISION = "ccbe24de388e287beb92dd393a335c376b350ac3"
LABELS = ["negative", "neutral", "positive"]
SOURCE_LABELS = {"0": "negative", "1": "positive", "2": "neutral"}


def read_csv(path, source_labels=False):
    with Path(path).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"text", "label"} <= set(reader.fieldnames):
            raise ValueError("CSV must contain text and label columns")
        rows = []
        for row in reader:
            text = " ".join((row["text"] or "").split())
            label = row["label"]
            if source_labels:
                if label not in SOURCE_LABELS:
                    raise ValueError(f"Unknown source label: {label!r}")
                label = SOURCE_LABELS[label]
            if label not in LABELS:
                raise ValueError(f"Unknown sentiment: {label!r}")
            rows.append({"text": text, "label": label})
    return rows


def clean_rows(rows):
    groups = {}
    for row in rows:
        text = " ".join(re.sub(r"https?://\S+", "", row["text"]).split())
        if row["label"] not in LABELS:
            raise ValueError(f"Unknown sentiment: {row['label']!r}")
        if text:
            groups.setdefault(text.casefold(), []).append(dict(row, text=text))
    # Drop conflicting annotations within a split; retain one copy otherwise.
    return [group[0] for group in groups.values()
            if len({row["label"] for row in group}) == 1]


def prepare_splits(train, test):
    train = clean_rows(train)
    test = clean_rows(test)
    train_texts = {row["text"].casefold() for row in train}
    test = [row for row in test if row["text"].casefold() not in train_texts]
    if not train or not test:
        raise ValueError("Both splits must contain data after cleaning")
    return train, test


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prepared():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    splits = []
    for name in ("train", "test"):
        path = DATA / f"{name}.csv"
        if file_hash(path) != manifest["sha256"][name]:
            raise ValueError("Prepared data changed; rerun prepare_data.py")
        splits.append(read_csv(path))
    if prepare_splits(*splits) != tuple(splits):
        raise ValueError("Prepared splits contain empty, conflicting or overlapping rows")
    return *splits, manifest
