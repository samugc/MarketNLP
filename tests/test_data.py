import pytest

from marketnlp.data import prepare_splits, read_csv, write_csv


def test_source_labels_and_unicode_roundtrip(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text('text,label\n"Profit, €42m",1\nLosses,0\nUnchanged,2\n', encoding="utf-8")
    rows = read_csv(source, source_labels=True)
    assert [row["label"] for row in rows] == ["positive", "negative", "neutral"]
    assert rows[0]["text"] == "Profit, €42m"
    write_csv(tmp_path / "prepared.csv", rows)
    assert read_csv(tmp_path / "prepared.csv") == rows


@pytest.mark.parametrize("content", ["text,label\nx,9\n", "text,label\nx,\n", "wrong,label\nx,1\n"])
def test_rejects_bad_csv(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_csv(path, source_labels=True)


def test_cleaning_split_is_reproducible_and_has_no_text_overlap():
    train = [{"text": "  Profit   rises https://t.co/abc ", "label": "positive"},
             {"text": "profit rises https://t.co/def", "label": "positive"},
             {"text": "ambiguous", "label": "positive"},
             {"text": "AMBIGUOUS", "label": "negative"},
             {"text": " ", "label": "neutral"}]
    test = [{"text": "PROFIT RISES https://t.co/ghi", "label": "negative"},
            {"text": "No change", "label": "neutral"}]
    expected = ([{"text": "Profit rises", "label": "positive"}],
                [{"text": "No change", "label": "neutral"}])
    assert prepare_splits(train, test) == expected
    assert prepare_splits(train, test) == expected
    assert train[0]["text"] == "  Profit   rises https://t.co/abc "


def test_empty_test_after_overlap_is_rejected():
    rows = [{"text": "same", "label": "neutral"}]
    with pytest.raises(ValueError):
        prepare_splits(rows, rows)


def test_prepared_data_hash_detects_changed_labels(tmp_path, monkeypatch):
    import json
    import marketnlp.data as data

    monkeypatch.setattr(data, "DATA", tmp_path)
    train = [{"text": "profit", "label": "positive"}]
    test = [{"text": "loss", "label": "negative"}]
    write_csv(tmp_path / "train.csv", train)
    write_csv(tmp_path / "test.csv", test)
    manifest = {"sha256": {name: data.file_hash(tmp_path / f"{name}.csv")
                           for name in ("train", "test")}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert data.load_prepared() == (train, test, manifest)
    write_csv(tmp_path / "test.csv", [{"text": "loss", "label": "positive"}])
    with pytest.raises(ValueError, match="changed"):
        data.load_prepared()
