import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from marketnlp.data import load_prepared
from marketnlp.finbert import MODEL, REVISION, label_mapping, predict
from marketnlp.metrics import save_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained FinBERT without fine-tuning")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    _, test, manifest = load_prepared()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION)
    print(f"FinBERT: {len(test)} texts, device={device}, labels={model.config.id2label}", flush=True)
    predictions = predict([row["text"] for row in test], model, tokenizer, device, args.batch_size)
    save_result("finbert", [row["label"] for row in test], predictions, manifest, {
        "model": MODEL, "revision": REVISION, "id2label": label_mapping(model.config.id2label),
        "device": device, "batch_size": args.batch_size, "max_length": 512, "fine_tuned_here": False,
    })


if __name__ == "__main__":
    main()
