from types import SimpleNamespace

import pytest
import torch

from marketnlp.finbert import label_mapping, predict


def test_label_mapping_uses_config_not_dataset_integer_order():
    assert label_mapping({"0": "positive", "1": "negative", "2": "neutral"}) == {
        0: "positive", 1: "negative", 2: "neutral"
    }
    with pytest.raises(ValueError):
        label_mapping({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_inference_preserves_order_and_handles_final_short_batch():
    batches = []

    def tokenize(texts, **kwargs):
        batches.append(texts)
        assert kwargs == {"padding": True, "truncation": True,
                          "max_length": 512, "return_tensors": "pt"}
        return {"input_ids": torch.tensor([[int(text)] for text in texts])}

    class Model(torch.nn.Module):
        config = SimpleNamespace(id2label={0: "positive", 1: "negative", 2: "neutral"})

        def forward(self, input_ids):
            assert not self.training
            assert not torch.is_grad_enabled()
            return SimpleNamespace(logits=torch.nn.functional.one_hot(input_ids[:, 0], 3).float())

    assert predict(["2", "0", "1"], Model(), tokenize, "cpu", batch_size=2) == [
        "neutral", "positive", "negative"
    ]
    assert batches == [["2", "0"], ["1"]]
    with pytest.raises(ValueError):
        predict(["0"], Model(), tokenize, "cpu", batch_size=0)
