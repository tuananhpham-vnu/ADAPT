"""Retriever encoding shared by attack and defense experiments."""
def encode(model, tokenizer, text: str, device: str):
    """Encode một câu văn thành vector bằng retriever đã nạp qua `algo.utils.load_models`."""
    import torch

    tokenized = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**tokenized)
    return output.pooler_output.squeeze(0)

