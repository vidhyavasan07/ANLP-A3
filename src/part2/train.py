import argparse
import json
import os
import random
import statistics
from pathlib import Path

import torch
from torch import nn
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from .models import build_model


MODEL_NAMES = ("mlp", "gcn", "sage", "gat", "bigcn")
FEATURE_NAMES = ("profile", "spacy", "bert", "content")
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


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


def train_one(model_name, feature, seed, train_loader, validation_loader, test_loader, in_channels, args, device, hub_api=None, group="architecture"):
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
    checkpoint_path = checkpoint_dir / f"{feature}_{model_name}_seed{seed}.pt"
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
                    "feature": feature,
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
        path_in_repo = f"{group}/{feature}/{model_name}/seed_{seed}.pt"
        hub_api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=path_in_repo,
            repo_id=args.hf_repo_id,
            repo_type="model",
        )
        test_metrics["hf_file"] = f"https://huggingface.co/{args.hf_repo_id}/blob/main/{path_in_repo}"
    return test_metrics


def summarize(runs):
    return {
        metric: {
            "mean": statistics.mean(run[metric] for run in runs),
            "std": statistics.stdev(run[metric] for run in runs) if len(runs) > 1 else 0.0,
        }
        for metric in METRIC_NAMES
    }


def make_loaders(feature, args):
    train_dataset = UPFD(root=args.data_root, name="gossipcop", feature=feature, split="train")
    validation_dataset = UPFD(root=args.data_root, name="gossipcop", feature=feature, split="val")
    test_dataset = UPFD(root=args.data_root, name="gossipcop", feature=feature, split="test")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return train_loader, validation_loader, test_loader, train_dataset[0].x.shape[-1]


def model_runs(model_name, feature, loaders, in_channels, args, device, hub_api, group):
    train_loader, validation_loader, test_loader = loaders
    runs = []
    for seed in args.seeds:
        runs.append(
            train_one(
                model_name,
                feature,
                seed,
                train_loader,
                validation_loader,
                test_loader,
                in_channels,
                args,
                device,
                hub_api,
                group,
            )
        )
    return {"runs": runs, "summary": summarize(runs)}


def config_from_args(args):
    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }


def load_matching_architecture_results(path, args):
    if path is None or not Path(path).is_file():
        return None
    source = json.loads(Path(path).read_text())
    if source.get("dataset") != "GossipCop" or source.get("feature") != "bert":
        return None
    if source.get("seeds") != args.seeds or source.get("config") != config_from_args(args):
        return None
    models = source.get("models", {})
    if any(name not in models for name in MODEL_NAMES):
        return None
    for entry in models.values():
        if len(entry.get("runs", [])) != len(args.seeds):
            return None
        if any(any(metric not in run for metric in METRIC_NAMES) for run in entry["runs"]):
            return None
    return source


def fresh_submission_results(path):
    if path.is_file():
        try:
            result = json.loads(path.read_text())
            if "part_a" in result and "part_b" in result:
                return result
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "part_a": {name: {metric: None for metric in METRIC_NAMES} for name in MODEL_NAMES},
        "part_b": {name: {metric: None for metric in METRIC_NAMES} for name in FEATURE_NAMES},
    }


def mean_only(entry):
    return {metric: entry["summary"][metric]["mean"] for metric in METRIC_NAMES}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/part2")
    parser.add_argument("--experiment", choices=("all", "architecture", "ablation"), default="all")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--features", nargs="+", choices=FEATURE_NAMES, default=list(FEATURE_NAMES))
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
    parser.add_argument("--output", default="output/part2/results.json")
    parser.add_argument("--details-output", default="output/part2/training_details.json")
    parser.add_argument("--architecture-results")
    parser.add_argument("--hf-repo-id", default=os.environ.get("HF_REPO_ID"))
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--ta", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_path = Path(args.output)
    if args.ta:
        output_path = output_path.with_name("results_ta.json")
    details_path = Path(args.details_output)
    hub_api = None
    if args.hf_repo_id:
        from huggingface_hub import HfApi

        hub_api = HfApi()
        hub_api.create_repo(repo_id=args.hf_repo_id, repo_type="model", private=args.hf_private, exist_ok=True)

    submitted = fresh_submission_results(output_path)
    details = {
        "dataset": "GossipCop",
        "seeds": args.seeds,
        "config": config_from_args(args),
        "architecture_comparison": {},
        "feature_ablation": {},
    }

    bert_loaders = None
    bert_in_channels = None
    architecture_sage = None
    external_architecture = None
    if args.experiment == "ablation":
        external_architecture = load_matching_architecture_results(args.architecture_results, args)
        if external_architecture is not None:
            for model_name in MODEL_NAMES:
                entry = external_architecture["models"][model_name]
                details["architecture_comparison"][model_name] = entry
                submitted["part_a"][model_name] = {metric: entry["summary"][metric]["mean"] for metric in METRIC_NAMES}
            architecture_sage = external_architecture["models"]["sage"]

    if args.experiment in ("all", "architecture"):
        bert_train, bert_validation, bert_test, bert_in_channels = make_loaders("bert", args)
        bert_loaders = (bert_train, bert_validation, bert_test)
        for model_name in args.models:
            entry = model_runs(model_name, "bert", bert_loaders, bert_in_channels, args, device, hub_api, "architecture_comparison")
            details["architecture_comparison"][model_name] = entry
            submitted["part_a"][model_name] = mean_only(entry)
            if model_name == "sage":
                architecture_sage = entry

    if args.experiment in ("all", "ablation"):
        reused_sage = architecture_sage
        for feature in args.features:
            if feature == "bert" and reused_sage is not None:
                entry = reused_sage
            else:
                if feature == "bert" and bert_loaders is not None:
                    loaders = bert_loaders
                    in_channels = bert_in_channels
                else:
                    train_loader, validation_loader, test_loader, in_channels = make_loaders(feature, args)
                    loaders = (train_loader, validation_loader, test_loader)
                entry = model_runs("sage", feature, loaders, in_channels, args, device, hub_api, "feature_ablation")
            details["feature_ablation"][feature] = entry
            submitted["part_b"][feature] = mean_only(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(submitted, indent=2) + "\n")
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(details, indent=2) + "\n")


if __name__ == "__main__":
    main()
