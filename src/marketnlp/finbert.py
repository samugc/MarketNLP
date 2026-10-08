from marketnlp.data import LABELS

MODEL = "ProsusAI/finbert"
REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"


def label_mapping(id2label):
    mapping = {int(key): value.lower() for key, value in id2label.items()}
    if set(mapping) != {0, 1, 2} or set(mapping.values()) != set(LABELS):
        raise ValueError(f"Unexpected model labels: {id2label}")
    return mapping


def predict(texts, model, tokenizer, device, batch_size=32):
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    mapping = label_mapping(model.config.id2label)
    model.to(device)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[start:start + batch_size], padding=True,
                               truncation=True, max_length=512, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            indices = model(**inputs).logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(mapping[index] for index in indices)
    return predictions
