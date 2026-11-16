import argparse
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

try:
    from .char_tokenizer import CharacterTokenizer
except ImportError:
    from char_tokenizer import CharacterTokenizer

try:
    from rapidfuzz.distance import Levenshtein as _lev
except ImportError:
    _lev = None

NOISE = frozenset("!'+,-./:;<=>?[]_{}")
PLAIN = "abcdefghijklmnopqrstuvwxyz "
UNBOUNDED = 10 ** 7
MODEL_SPECS = {
    "transformer": ("transformer", "DecoderOnlyTransformer"),
    "mamba": ("mamba", "MambaLanguageModel"),
    "deltanet": ("deltanet", "DeltaNetLanguageModel"),
    "hybrid": ("hybrid", "HybridLanguageModel"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("test", "heldout"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["mamba"], choices=sorted(MODEL_SPECS))
    parser.add_argument("--hf-repo-id", default=os.environ.get("HF_REPO", "vidhyavasan/anlp-a3"))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/part1"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--flops-len", type=int, default=1024)
    return parser.parse_args()


def edit_distance(a, b):
    if _lev is not None:
        return _lev.distance(a, b)
    vocab = {}
    ai = np.array([vocab.setdefault(x, len(vocab)) for x in a], dtype=np.int64)
    bi = np.array([vocab.setdefault(x, len(vocab)) for x in b], dtype=np.int64)
    prev = np.arange(len(bi) + 1)
    idx = np.arange(len(bi) + 1)
    for i in range(1, len(ai) + 1):
        cost = (bi != ai[i - 1]).astype(np.int64)
        base = np.empty(len(bi) + 1, dtype=np.int64)
        base[0] = i
        base[1:] = np.minimum(prev[1:] + 1, prev[:-1] + cost)
        prev = idx + np.minimum.accumulate(base - idx)
    return int(prev[-1])


def forward_flops(model, seq_len, name, model_config):
    per_token = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            per_token += 2 * m.in_features * m.out_features
        elif isinstance(m, torch.nn.Conv1d):
            per_token += 2 * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0]
        elif hasattr(m, "d_state") and hasattr(m, "d_inner"):
            per_token += 7 * m.d_inner * m.d_state
        elif hasattr(m, "head_dim") and hasattr(m, "chunk"):
            per_token += m.n_heads * (6 * m.head_dim ** 2 + 8 * m.chunk * m.head_dim)
    total = per_token * seq_len
    if name == "transformer":
        total += model_config["n_layers"] * 2 * seq_len ** 2 * model_config["d_model"]
    return total


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_test_files(path):
    path = Path(path)
    for candidate in (path, path / "test", path / "test-data", path / "test_data"):
        cipher, text = candidate / "cipher.jsonl", candidate / "cleaned.jsonl"
        if cipher.is_file() and text.is_file():
            return cipher, text
    raise FileNotFoundError(f"cipher.jsonl and cleaned.jsonl not found under {path}")


def find_heldout_file(path):
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("cipher.jsonl", "heldout.jsonl"):
            if (path / name).is_file():
                return path / name
        files = sorted(path.glob("*.jsonl"))
        if len(files) == 1:
            return files[0]
    for suffix in (".jsonl", ".json"):
        candidate = path.with_name(path.name + suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"heldout ciphertext file not found at {path}")


def import_model_class(module_name, class_name):
    package = __package__
    try:
        module = importlib.import_module(f"{package}.{module_name}" if package else module_name)
    except ImportError:
        module = importlib.import_module(module_name)
    return getattr(module, class_name)


def locate_checkpoint(name, args):
    if args.checkpoint_dir:
        base = Path(args.checkpoint_dir)
        for candidate in (base / name / "model.pt", base / "model.pt"):
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(f"no model.pt for {name} under {base}")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=args.hf_repo_id, filename=f"{name}/model.pt", token=os.environ.get("HF_TOKEN")
    )


def build_model(name, args, device):
    path = locate_checkpoint(name, args)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    module_name, class_name = MODEL_SPECS[name]
    model = import_model_class(module_name, class_name)(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = CharacterTokenizer.from_state_dict(checkpoint["tokenizer"])
    limit = model.max_seq_len if name == "transformer" else UNBOUNDED
    model.max_seq_len = limit
    return model, tokenizer, checkpoint["model_config"], limit


def make_predictor(model, tokenizer, limit, device):
    allowed = torch.full((model.vocab_size,), float("-inf"), device=device)
    allowed[[tokenizer.encode(ch)[0] for ch in PLAIN]] = 0.0

    @torch.no_grad()
    def predict(ciphertext):
        core = "".join(ch for ch in ciphertext if ch not in NOISE)
        ids = tokenizer.encode(core)[:limit]
        if not ids:
            return []
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        return (model(tensor)[0] + allowed).argmax(-1).tolist()

    return predict


def evaluate_model(name, model, tokenizer, model_config, limit, cipher_rows, text_rows, args, device, predictions_path):
    vocab = model.vocab_size
    predict = make_predictor(model, tokenizer, limit, device)
    tp, pred_n, ref_n = np.zeros(vocab), np.zeros(vocab), np.zeros(vocab)
    total_correct = total_chars = lev_sum = beyond = 0
    word_hit = word_total = word_lev = exact = 0
    per_example = []
    buckets = {"<=127": [0, 0], "128-1023": [0, 0], ">=1024": [0, 0]}
    start = time.time()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as out:
        for index, (crow, trow) in enumerate(zip(cipher_rows, text_rows)):
            if crow.get("id") != trow.get("id"):
                raise ValueError("ciphertext and plaintext test rows are not aligned")
            ref = trow["text"]
            pred_ids = predict(crow["ciphertext"])
            pred = tokenizer.decode(pred_ids)
            ref_ids = np.array(tokenizer.encode(ref), dtype=np.int64)
            pa = np.array(pred_ids, dtype=np.int64)
            n = min(len(pa), len(ref_ids))
            hit = pa[:n] == ref_ids[:n]
            correct = int(hit.sum())
            total_correct += correct
            total_chars += len(ref_ids)
            beyond += max(len(ref) - limit, 0)
            tp += np.bincount(ref_ids[:n][hit], minlength=vocab)
            pred_n += np.bincount(pa[:n], minlength=vocab)
            ref_n += np.bincount(ref_ids, minlength=vocab)
            per_example.append(correct / max(len(ref_ids), 1))
            key = "<=127" if len(ref) <= 127 else "128-1023" if len(ref) < 1024 else ">=1024"
            buckets[key][0] += correct
            buckets[key][1] += len(ref)
            lev_sum += edit_distance(pred, ref)
            exact += pred == ref
            pred_words, ref_words = pred.split(), ref.split()
            word_hit += sum(x == y for x, y in zip(pred_words, ref_words))
            word_total += len(ref_words)
            word_lev += edit_distance(pred_words, ref_words)
            out.write(json.dumps({"id": crow["id"], "prediction": pred, "reference": ref}, ensure_ascii=False) + "\n")
            if (index + 1) % 100 == 0:
                print(f"{name}: {index + 1}/{len(text_rows)}", flush=True)
    precision = np.divide(tp, pred_n, out=np.zeros(vocab), where=pred_n > 0)
    recall = np.divide(tp, ref_n, out=np.zeros(vocab), where=ref_n > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(vocab), where=(precision + recall) > 0)
    seen = ref_n > 0
    count = len(text_rows)
    return {
        "examples": count,
        "char_accuracy": total_correct / total_chars,
        "mean_example_char_accuracy": float(np.mean(per_example)),
        "word_accuracy": word_hit / max(word_total, 1),
        "exact_match": exact / count,
        "cer": lev_sum / total_chars,
        "wer": word_lev / max(word_total, 1),
        "levenshtein_mean": lev_sum / count,
        "precision_macro": float(precision[seen].mean()),
        "recall_macro": float(recall[seen].mean()),
        "f1_macro": float(f1[seen].mean()),
        "parameters": sum(p.numel() for p in model.parameters()),
        f"flops_forward_len{args.flops_len}": forward_flops(model, args.flops_len, name, model_config),
        "chars_beyond_context": beyond / total_chars,
        "char_accuracy_by_length": {k: v[0] / v[1] for k, v in buckets.items() if v[1]},
        "eval_seconds": round(time.time() - start, 1),
    }


def run_heldout(args, device):
    if len(args.models) != 1:
        raise ValueError("heldout mode needs exactly one model")
    if args.output is None:
        raise ValueError("--output is required in heldout mode")
    rows = read_jsonl(find_heldout_file(args.data))
    if args.limit:
        rows = rows[: args.limit]
    name = args.models[0]
    model, tokenizer, _, limit = build_model(name, args, device)
    predict = make_predictor(model, tokenizer, limit, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            text = tokenizer.decode(predict(row["ciphertext"]))
            out.write(json.dumps({"id": row["id"], "text": text}, ensure_ascii=False) + "\n")
            if (index + 1) % 100 == 0:
                print(f"{name}: {index + 1}/{len(rows)}", flush=True)
    print(f"wrote {len(rows)} rows to {args.output}")


def run_test(args, device):
    cipher_path, text_path = find_test_files(args.data)
    cipher_rows, text_rows = read_jsonl(cipher_path), read_jsonl(text_path)
    if args.limit:
        cipher_rows, text_rows = cipher_rows[: args.limit], text_rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.models:
        try:
            model, tokenizer, model_config, limit = build_model(name, args, device)
            metrics = evaluate_model(
                name, model, tokenizer, model_config, limit, cipher_rows, text_rows, args, device,
                args.output_dir / f"predictions_{name}.jsonl",
            )
        except Exception as error:
            print(f"{name}: skipped ({type(error).__name__}: {error})", flush=True)
            results[name] = {"error": f"{type(error).__name__}: {error}"}
            continue
        results[name] = metrics
        (args.output_dir / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(name)
        print(json.dumps(metrics, indent=2), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    done = [name for name, value in results.items() if "error" not in value]
    if not done:
        raise SystemExit(1)
    keys = [
        "char_accuracy", "mean_example_char_accuracy", "word_accuracy", "exact_match", "cer", "wer",
        "levenshtein_mean", "precision_macro", "recall_macro", "f1_macro", "parameters", "chars_beyond_context",
    ]
    print(f"{'metric':30s}" + "".join(f"{name:>14s}" for name in done))
    for key in keys:
        print(f"{key:30s}" + "".join(f"{results[name][key]:>14.4f}" for name in done))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "heldout":
        run_heldout(args, device)
    else:
        run_test(args, device)


if __name__ == "__main__":
    main()