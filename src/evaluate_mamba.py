import argparse
import json
from pathlib import Path

import torch

try:
    from .char_tokenizer import CharacterTokenizer
    from .mamba import MambaLanguageModel
except ImportError:
    from char_tokenizer import CharacterTokenizer
    from mamba import MambaLanguageModel


def find_test_files(test_data):
    test_data = Path(test_data)
    if test_data.is_file():
        raise ValueError("--test-data must be a directory")
    candidates = [test_data]
    if test_data.name not in {"test", "test-data", "test_data", "test_data1"}:
        candidates.append(test_data / "test")
        candidates.append(test_data / "test-data")
        candidates.append(test_data / "test_data")
        candidates.append(test_data / "test_data1")
    for candidate in candidates:
        cipher_path = candidate / "cipher.jsonl"
        text_path = candidate / "cleaned.jsonl"
        if cipher_path.is_file() and text_path.is_file():
            return cipher_path, text_path
    raise FileNotFoundError(
        f"Could not find cipher.jsonl and cleaned.jsonl in {test_data}, {test_data / 'test'}, "
        f"{test_data / 'test-data'}, {test_data / 'test_data'}, or {test_data / 'test_data1'}"
    )


def generate(model, tokenizer, ciphertext, max_seq_len, device):
    cipher_ids = tokenizer.encode(ciphertext)
    prefix = [tokenizer.bos_id, *cipher_ids, tokenizer.sep_id]
    prefix = prefix[:max_seq_len]
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    output = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_seq_len - len(prefix)):
            logits = model(input_ids)
            next_id = int(logits[0, -1].argmax().item())
            if next_id == tokenizer.eos_id:
                break
            output.append(next_id)
            input_ids = torch.cat(
                (input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)), dim=1
            )
    return tokenizer.decode(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", "--data-dir", dest="test_data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = MambaLanguageModel(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = CharacterTokenizer.from_state_dict(checkpoint["tokenizer"])
    cipher_path, text_path = find_test_files(args.test_data)

    predictions = []
    exact_matches = 0
    total_characters = 0
    correct_characters = 0
    with open(cipher_path, encoding="utf-8") as cipher_file, open(
        text_path, encoding="utf-8"
    ) as text_file:
        for cipher_line, text_line in zip(cipher_file, text_file):
            cipher_row = json.loads(cipher_line)
            text_row = json.loads(text_line)
            if cipher_row.get("id") != text_row.get("id"):
                raise ValueError("Ciphertext and plaintext test rows are not aligned")
            prediction = generate(
                model, tokenizer, cipher_row["ciphertext"], config["max_seq_len"], device
            )
            reference = text_row["text"]
            exact_matches += prediction == reference
            total_characters += len(reference)
            correct_characters += sum(a == b for a, b in zip(prediction, reference))
            predictions.append({"id": cipher_row["id"], "prediction": prediction, "reference": reference})

    if len(predictions) == 0:
        raise ValueError("Test split is empty")
    metrics = {
        "examples": len(predictions),
        "exact_match_accuracy": exact_matches / len(predictions),
        "character_accuracy": correct_characters / total_characters if total_characters else 0.0,
        "checkpoint_epoch": checkpoint.get("epoch"),
    }
    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as output_file:
            for row in predictions:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
