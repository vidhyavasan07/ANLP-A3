import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from torch import nn
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from .models import build_model


MODEL_NAMES = ("mlp", "gcn", "sage", "gat", "bigcn")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_metrics(predictions, labels):
    predictions = torch.cat(predictions).long()
    labels = torch.cat(labels).long()
    true_positive = ((predictions == 1) & (labels == 1)).sum().item()
    true_negative = ((predictions == 0) & (labels == 0)).sum().item()
    false_positive = ((predictions == 1) & (labels == 0)).sum().item()
    false_negative = ((predictions == 0) & (labels == 1)).sum().item()
    accuracy = (true_positive + true_negative) / max(labels.numel(), 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses = []
    predictions = []
    labels = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for data in loader:
            data = data.to(device)
            targets = data.y.reshape(-1).long()
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(data.x.float(), data.edge_index, data.batch)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.detach().item())
            predictions.append(logits.argmax(dim=-1).detach().cpu())
            labels.append(targets.detach().cpu())
    metrics = binary_metrics(predictions, labels)
    metrics["loss"] = statistics.mean(losses) if losses else 0.0
    return metrics


def train_one(model_name, seed, train_loader, validation_loader, test_loader, in_channels, args, device, hub_api=None):
    set_seed(seed)
    model = build_model(
        model_name,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_classes=2,
        dropout=args.dropout,
        num_layers=args.num_layers,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_seed{seed}.pt"
    best_validation_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        run_epoch(model, train_loader, device, criterion, optimizer)
        validation_metrics = run_epoch(model, validation_loader, device, criterion)
        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_f1": best_validation_f1,
                    "in_channels": in_channels,
                    "hidden_channels": args.hidden_channels,
                    "num_layers": args.num_layers,
                    "heads": args.heads,
                    "dropout": args.dropout,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(model, test_loader, device, criterion)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_validation_f1"] = best_validation_f1
    test_metrics["checkpoint"] = str(checkpoint_path)
    if hub_api is not None:
        path_in_repo = f"architecture_comparison/{model_name}/seed_{seed}.pt"
        hub_api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=path_in_repo,
            repo_id=args.hf_repo_id,
            repo_type="model",
        )
        test_metrics["hf_file"] = f"https://huggingface.co/{args.hf_repo_id}/blob/main/{path_in_repo}"
    return test_metrics


def summarize(runs):
    metric_names = ("accuracy", "precision", "recall", "f1")
    summary = {}
    for name in metric_names:
        values = [run[name] for run in runs]
        summary[name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/part2")
    parser.add_argument("--feature", choices=("bert",), default="bert")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default="models/part2")
    parser.add_argument("--output", default="output/part2/architecture_comparison.json")
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-private", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    hub_api = None
    if args.hf_repo_id:
        from huggingface_hub import HfApi

        hub_api = HfApi()
        hub_api.create_repo(repo_id=args.hf_repo_id, repo_type="model", private=args.hf_private, exist_ok=True)
    train_dataset = UPFD(root=args.data_root, name="gossipcop", feature=args.feature, split="train")
    validation_dataset = UPFD(root=args.data_root, name="gossipcop", feature=args.feature, split="val")
    test_dataset = UPFD(root=args.data_root, name="gossipcop", feature=args.feature, split="test")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    in_channels = train_dataset[0].x.shape[-1]
    results = {
        "dataset": "GossipCop",
        "feature": args.feature,
        "seeds": args.seeds,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "models": {},
    }
    for model_name in args.models:
        runs = []
        for seed in args.seeds:
            runs.append(
                train_one(
                    model_name,
                    seed,
                    train_loader,
                    validation_loader,
                    test_loader,
                    in_channels,
                    args,
                    device,
                    hub_api,
                )
            )
        results["models"][model_name] = {"runs": runs, "summary": summarize(runs)}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
