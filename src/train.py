"""Train the character-level cipher-to-text transformer.

Typical Kaggle usage from the repository root::

    python src/train.py \
        --data-dir /kaggle/input/YOUR-DATASET/data \
        --output-dir /kaggle/working/a3-transformer \
        --hf-repo-id YOUR-HF-USERNAME/a3-transformer \
        --wandb-project a3-transformer

The training data directory should contain ``train/cipher.jsonl`` and
``train/cleaned.jsonl``. A sibling ``validation`` directory is used when
present. Cloud uploads are optional; credentials are read from environment
variables or Kaggle Secrets and are never written to the checkpoint.
"""

import argparse
import json
import os
import random
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

try:  # Support both ``python src/train.py`` and ``python -m src.train``.
    from .char_tokenizer import CharacterTokenizer
    from .sequence_dataset import PairedTextDataset, collate_sequences
    from .transformer import DecoderOnlyTransformer
except ImportError:
    from char_tokenizer import CharacterTokenizer
    from sequence_dataset import PairedTextDataset, collate_sequences
    from transformer import DecoderOnlyTransformer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing train/ and optionally validation/; auto-detected if omitted.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/a3-transformer"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Use only this many training rows; 0 means all rows.")
    parser.add_argument("--max-validation-samples", type=int, default=0,
                        help="Use only this many validation rows; 0 means all rows.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-repo-id", help="Optional Hugging Face model repo, e.g. user/a3-transformer.")
    parser.add_argument("--hf-private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", help="Optional Weights & Biases project name.")
    parser.add_argument("--wandb-run-name", help="Optional W&B run name.")
    return parser.parse_args()


def find_split(root, split):

    root = Path(root)
    if not root.exists():
        return None
    split_directories = {
        "train": {"train", "train_data", "train-data"},
        "validation": {"validation", "validation_data", "validation-data", "val"},
    }.get(split, {split})
    for cipher_path in sorted(root.rglob("cipher.jsonl")):
        if cipher_path.parent.name not in split_directories:
            continue
        cleaned_path = cipher_path.with_name("cleaned.jsonl")
        if cleaned_path.is_file():
            return cipher_path, cleaned_path
    return None


def locate_data(data_dir):
    
    roots = []
    if data_dir is not None:
        roots.append(Path(data_dir))
    roots.extend((Path("/kaggle/input"), Path(__file__).resolve().parents[1] / "data"))

    # Remove duplicate roots while preserving the requested path first.
    unique_roots = []
    seen = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)

    for root in unique_roots:
        train_pair = find_split(root, "train")
        if train_pair is not None:
            # Prefer validation under the same data root as the training split.
            validation_pair = find_split(train_pair[0].parent.parent, "validation")
            if validation_pair is None:
                validation_pair = find_split(root, "validation")
            return train_pair, validation_pair

    expected = "\n".join(f"  - {root}" for root in unique_roots)
    raise FileNotFoundError(
        "Could not find train/cipher.jsonl and train/cleaned.jsonl. "
        "Attach the dataset in Kaggle's Add Input panel or pass --data-dir. Searched:\n"
        f"{expected}"
    )


def limit_dataset(dataset, limit):
    if limit and limit > 0:
        return Subset(dataset, range(min(limit, len(dataset))))
    return dataset


def get_secret(name):
    """Read a token from the environment, or from Kaggle Secrets if available."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            _, loss = model(input_ids, targets)
            total_loss += loss.item()
            total_batches += 1
    model.train()
    return total_loss / total_batches if total_batches else None


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.n_heads < 1 or args.n_layers < 1:
        raise ValueError("epochs, batch-size, n-heads, and n-layers must be positive")
    if args.log_every < 1 or args.num_workers < 0:
        raise ValueError("log-every must be positive and num-workers cannot be negative")
    if args.max_seq_len < 4:
        raise ValueError("max-seq-len must be at least 4")
    if args.d_model % args.n_heads != 0:
        raise ValueError("d-model must be divisible by n-heads")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    hf_api = None
    if args.hf_repo_id:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise RuntimeError("Install Hugging Face Hub first: pip install huggingface_hub") from error
        hf_token = get_secret("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "Hugging Face upload was requested but HF_TOKEN is unavailable. "
                "Add a write token in Kaggle Add-ons > Secrets and enable notebook access."
            )
        hf_api = HfApi(token=hf_token)

    train_pair, validation_pair = locate_data(args.data_dir)
    train_cipher, train_text = train_pair
    print(f"Training ciphertext: {train_cipher}")
    print(f"Training text:       {train_text}")
    if validation_pair:
        print(f"Validation files:    {validation_pair[0]} and {validation_pair[1]}")
    else:
        print("No validation split found; validation loss will be skipped.")

    # Build the vocabulary from training text only to avoid validation leakage.
    tokenizer = CharacterTokenizer.from_jsonl(train_cipher, train_text)
    train_dataset = PairedTextDataset(
        train_cipher, train_text, tokenizer, args.max_seq_len
    )
    train_dataset = limit_dataset(train_dataset, args.max_train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(collate_sequences, pad_id=tokenizer.pad_id),
    )

    validation_loader = None
    if validation_pair:
        validation_dataset = PairedTextDataset(
            validation_pair[0], validation_pair[1], tokenizer, args.max_seq_len
        )
        validation_dataset = limit_dataset(
            validation_dataset, args.max_validation_samples
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=partial(collate_sequences, pad_id=tokenizer.pad_id),
        )

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
        "max_seq_len": args.max_seq_len,
        "dropout": args.dropout,
        "pad_id": tokenizer.pad_id,
    }
    model = DecoderOnlyTransformer(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    run = None
    if args.wandb_project:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("Install W&B first: pip install wandb") from error
        wandb_key = get_secret("WANDB_API_KEY")
        if not wandb_key:
            raise RuntimeError(
                "W&B was requested but WANDB_API_KEY is unavailable. Add it in Kaggle "
                "Add-ons > Secrets and enable notebook access."
            )
        wandb.login(key=wandb_key)
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **model_config,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
            },
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    tokenizer_path = args.output_dir / "tokenizer.json"
    checkpoint_path = args.output_dir / "model.pt"
    config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    tokenizer_path.write_text(
        json.dumps(tokenizer.state_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    global_step = 0
    history = []
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            epoch_batches = 0

            for batch_index, (input_ids, targets) in enumerate(train_loader, start=1):
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                _, loss = model(input_ids, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                global_step += 1
                epoch_batches += 1
                running_loss += loss.item()
                if run is not None and global_step % args.log_every == 0:
                    run.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)
                if batch_index % args.log_every == 0:
                    print(
                        f"epoch {epoch}/{args.epochs}, batch {batch_index}/"
                        f"{len(train_loader)}, loss {loss.item():.4f}"
                    )

            train_loss = running_loss / max(epoch_batches, 1)
            validation_loss = (
                evaluate(model, validation_loader, device)
                if validation_loader is not None
                else None
            )
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
            history.append(metrics)
            print(
                f"epoch {epoch}: train loss={train_loss:.4f}"
                + (
                    f", validation loss={validation_loss:.4f}"
                    if validation_loss is not None
                    else ""
                )
            )
            if run is not None:
                run.log(
                    {
                        "epoch": epoch,
                        "train/epoch_loss": train_loss,
                        **(
                            {"validation/loss": validation_loss}
                            if validation_loss is not None
                            else {}
                        ),
                    },
                    step=global_step,
                )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model_config,
                    "tokenizer": tokenizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "metrics": metrics,
                },
                checkpoint_path,
            )
            (args.output_dir / "metrics.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

        print(f"Saved model checkpoint and metadata in {args.output_dir}")

        if args.hf_repo_id:
            hf_api.create_repo(
                repo_id=args.hf_repo_id,
                repo_type="model",
                private=args.hf_private,
                exist_ok=True,
            )
            hf_api.upload_folder(
                folder_path=str(args.output_dir),
                repo_id=args.hf_repo_id,
                repo_type="model",
                token=hf_token,
                commit_message="Upload trained A3 transformer checkpoint",
            )
            print(f"Uploaded model files to https://huggingface.co/{args.hf_repo_id}")

        if run is not None:
            artifact = wandb.Artifact("a3-transformer", type="model")
            artifact.add_file(str(checkpoint_path))
            artifact.add_file(str(config_path))
            artifact.add_file(str(tokenizer_path))
            run.log_artifact(artifact)
            print("Logged checkpoint artifact to Weights & Biases")
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
