"""Deterministic training and evaluation for the Houssein UTD-MHAD baseline."""

from __future__ import annotations

import copy
import json
import platform
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .model import HousseinBaseline


PAPER_PROTOCOL = {
    "optimizer": "RMSprop",
    "learning_rate": 9.8e-4,
    "weight_decay": 1.6e-4,
    "batch_size": 4,
    "scheduler": "ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-5)",
    "early_stopping_patience": 15,
    "max_epochs": 300,
    "table_2_dropouts": {"branch": 0.12, "classifier": 0.82},
}

REPRODUCTION_DEVIATIONS = [
    "The paper does not disclose its validation-subject split; this run freezes subjects 1,3,5 for development training and subject 7 for validation.",
    "The cached preprocessing uses proportional visual-to-IMU index mapping because released files have no timestamps; the paper's phrase 'same selected keypoint indices' is ambiguous for streams with different rates.",
    "The cached skeleton and inertial inputs are raw selected features; depth follows the paper-described OpenCV segmentation and HOG path.",
    "The paper's Fig. 5 shows Recognition BatchNorm before the first fully connected layer; this implementation includes it, while the paper does not disclose enough widths to match its reported parameter count.",
    "The paper's Fig. 2 code uses undisclosed filters0-filters5/lstm_units/dropout names that conflict with Table 2; no widths are inferred from that schematic.",
    "The paper reports dropout 0.12/0.82 in Table 2 but discusses 0.3/0.5 later; the Table 2 values are the primary frozen choice.",
    "The paper's 96.99% UTD-MHAD claim matches macro per-class recall (balanced accuracy) from its displayed class fractions; the valid 861-sample archive has 430 official test samples, so ordinary accuracy is reported separately.",
    "Final training joins all odd subjects and uses a recorded fixed epoch rule and fixed initial learning rate; validation-driven scheduling and early stopping apply only to development.",
]


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    max_epochs: int = 300
    learning_rate: float = 9.8e-4
    weight_decay: float = 1.6e-4
    scheduler_patience: int = 5
    early_stopping_patience: int = 15
    min_learning_rate: float = 1e-5
    branch_dropout: float = 0.12
    classifier_dropout: float = 0.82
    classifier_width: int = 128


class FeatureDataset(Dataset[tuple[Tensor, ...]]):
    """Indexed view of the precomputed literal-feature cache."""

    def __init__(self, cache_path: str | Path, indices: Iterable[int]) -> None:
        archive = np.load(cache_path)
        self.depth = torch.from_numpy(archive["depth"].astype(np.float32, copy=False))
        self.skeleton = torch.from_numpy(archive["skeleton"].astype(np.float32, copy=False))
        self.inertial = torch.from_numpy(archive["inertial"].astype(np.float32, copy=False))
        self.depth_mask = torch.from_numpy(archive["depth_mask"].astype(np.float32, copy=False))
        self.skeleton_mask = torch.from_numpy(archive["skeleton_mask"].astype(np.float32, copy=False))
        self.inertial_mask = torch.from_numpy(archive["inertial_mask"].astype(np.float32, copy=False))
        self.labels = torch.from_numpy(archive["actions"].astype(np.int64, copy=False) - 1)
        self.indices = np.asarray(tuple(indices), dtype=np.int64)
        if self.indices.size == 0:
            raise ValueError("dataset split is empty")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[Tensor, ...]:
        index = int(self.indices[position])
        return (
            self.depth[index],
            self.skeleton[index],
            self.inertial[index],
            self.depth_mask[index],
            self.skeleton_mask[index],
            self.inertial_mask[index],
            self.labels[index],
        )


def set_deterministic(seed: int) -> None:
    """Set the random sources used by the loader and model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _split_indices(cache_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(cache_path)
    subjects = archive["subjects"]
    development_train = np.flatnonzero(np.isin(subjects, (1, 3, 5)))
    development_valid = np.flatnonzero(subjects == 7)
    official_test = np.flatnonzero(np.isin(subjects, (2, 4, 6, 8)))
    if tuple(map(len, (development_train, development_valid, official_test))) != (323, 108, 430):
        raise ValueError("feature cache does not match the frozen UTD-MHAD protocol")
    return development_train, development_valid, official_test


def _loader(dataset: FeatureDataset, config: TrainingConfig, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _step(model: HousseinBaseline, batch: tuple[Tensor, ...], device: torch.device) -> tuple[Tensor, Tensor]:
    depth, skeleton, inertial, depth_mask, skeleton_mask, inertial_mask, labels = (
        value.to(device, non_blocking=True) for value in batch
    )
    logits, weights = model(depth, skeleton, inertial, depth_mask, skeleton_mask, inertial_mask)
    return torch.nn.functional.cross_entropy(logits, labels), logits


def _run_epoch(
    model: HousseinBaseline,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            loss, logits = _step(model, batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            labels = batch[-1].numpy()
            losses.append(float(loss.detach().cpu()))
            predictions.append(logits.detach().argmax(dim=1).cpu().numpy())
            targets.append(labels)
    return float(np.mean(losses)), np.concatenate(predictions), np.concatenate(targets)


def _metrics(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 27) -> dict[str, object]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": [
            {"class": index + 1, "support": int(support[index]), "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index])}
            for index in range(num_classes)
        ],
        "confusion_matrix": confusion.tolist(),
    }


def _train_fixed(
    cache_path: str | Path,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    seed: int,
    epochs: int,
    config: TrainingConfig,
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, object]:
    set_deterministic(seed)
    train_data = FeatureDataset(cache_path, train_indices)
    eval_data = FeatureDataset(cache_path, evaluation_indices)
    train_loader = _loader(train_data, config, shuffle=True, seed=seed)
    eval_loader = _loader(eval_data, config, shuffle=False, seed=seed)
    model = HousseinBaseline(
        branch_dropout=config.branch_dropout,
        classifier_dropout=config.classifier_dropout,
        classifier_width=config.classifier_width,
    ).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    started = time.perf_counter()
    train_loss = float("nan")
    for _ in range(epochs):
        train_loss, _, _ = _run_epoch(model, train_loader, device, optimizer)
    evaluation_loss, predictions, targets = _run_epoch(model, eval_loader, device, None)
    elapsed = time.perf_counter() - started
    result = {
        "seed": seed,
        "epochs": epochs,
        "train_loss": train_loss,
        "evaluation_loss": evaluation_loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        **_metrics(predictions, targets),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / f"final_seed{seed}.pt")
    return result


def _development_run(
    cache_path: str | Path,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    seed: int,
    config: TrainingConfig,
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, object]:
    set_deterministic(seed)
    train_data = FeatureDataset(cache_path, train_indices)
    validation_data = FeatureDataset(cache_path, validation_indices)
    train_loader = _loader(train_data, config, shuffle=True, seed=seed)
    validation_loader = _loader(validation_data, config, shuffle=False, seed=seed)
    model = HousseinBaseline(
        branch_dropout=config.branch_dropout,
        classifier_dropout=config.classifier_dropout,
        classifier_width=config.classifier_width,
    ).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=config.scheduler_patience, min_lr=config.min_learning_rate
    )
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_predictions, train_targets = _run_epoch(model, train_loader, device, optimizer)
        validation_loss, validation_predictions, validation_targets = _run_epoch(model, validation_loader, device, None)
        scheduler.step(validation_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": float(np.mean(train_predictions == train_targets)),
            "validation_accuracy": float(np.mean(validation_predictions == validation_targets)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("development run did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_loss, predictions, targets = _run_epoch(model, validation_loader, device, None)
    elapsed = time.perf_counter() - started
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_loss": best_loss,
        "validation_loss": validation_loss,
        "elapsed_seconds": elapsed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        **_metrics(predictions, targets),
        "history": history,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / f"development_seed{seed}.pt")
    return result


def run_paper_visible_diagnostic(
    cache_path: str | Path,
    output_dir: str | Path,
    *,
    seeds: tuple[int, ...] = (11,),
    training_config: TrainingConfig | None = None,
    max_epochs: int | None = None,
    classifier_width: int | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Run the paper-visible odd/even protocol as an explicitly leaky diagnostic.

    The even-subject official test set is used as validation each epoch.  This
    runner is intentionally separate from ``run_reproduction`` and its result
    must not be used as the clean final summary.
    """
    config = training_config or TrainingConfig()
    if max_epochs is not None:
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        config = replace(config, max_epochs=max_epochs)
    if classifier_width is not None:
        config = replace(config, classifier_width=classifier_width)
    if not seeds:
        raise ValueError("seeds must not be empty")
    cache_path = Path(cache_path)
    output_path = Path(output_dir)
    development_train, development_valid, official_test = _split_indices(cache_path)
    odd_subjects = np.concatenate((development_train, development_valid))
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    runs = [
        _development_run(
            cache_path,
            odd_subjects,
            official_test,
            seed=seed,
            config=config,
            device=selected_device,
            output_dir=output_path / f"seed{seed}",
        )
        for seed in seeds
    ]
    result = {
        "diagnostic_status": "diagnostic-only: test-set leakage",
        "dataset": "UTD-MHAD",
        "protocol": {
            "training_subjects": [1, 3, 5, 7],
            "validation_subjects": [2, 4, 6, 8],
            "training_count": len(odd_subjects),
            "validation_count": len(official_test),
        },
        "training_config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": str(selected_device),
            "device_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "CPU",
        },
        "runs": runs,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_reproduction(
    cache_path: str | Path,
    output_dir: str | Path,
    *,
    development_seeds: tuple[int, ...] = (11, 22, 33),
    final_seeds: tuple[int, ...] = (11, 22, 33, 44, 55),
    training_config: TrainingConfig | None = None,
    final_epochs: int | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Run frozen development selection, then fixed-epoch official testing."""
    config = training_config or TrainingConfig()
    cache_path = Path(cache_path)
    output_path = Path(output_dir)
    development_train, development_valid, official_test = _split_indices(cache_path)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    development = [
        _development_run(cache_path, development_train, development_valid, seed=seed, config=config, device=selected_device, output_dir=output_path / "development")
        for seed in development_seeds
    ]
    best_epochs = [int(item["best_epoch"]) for item in development]
    selected_epoch = int(np.median(best_epochs))
    training_epochs = selected_epoch if final_epochs is None else final_epochs
    if training_epochs < 1:
        raise ValueError("final_epochs must be positive")
    final = [
        _train_fixed(cache_path, np.concatenate((development_train, development_valid)), official_test, seed=seed, epochs=training_epochs, config=config, device=selected_device, output_dir=output_path / "final")
        for seed in final_seeds
    ]
    accuracies = np.asarray([float(item["accuracy"]) for item in final])
    balanced_accuracies = np.asarray([float(item["balanced_accuracy"]) for item in final])
    macro_f1 = np.asarray([float(item["macro_f1"]) for item in final])
    result = {
        "reproduction_status": "audited reconstruction, not exact reproduction",
        "dataset": "UTD-MHAD",
        "protocol": {
            "development_train_subjects": [1, 3, 5],
            "development_validation_subjects": [7],
            "official_train_subjects": [1, 3, 5, 7],
            "official_test_subjects": [2, 4, 6, 8],
            "development_count": len(development_train),
            "validation_count": len(development_valid),
            "official_train_count": len(development_train) + len(development_valid),
            "official_test_count": len(official_test),
        },
        "paper_protocol": PAPER_PROTOCOL,
        "reproduction_deviations": REPRODUCTION_DEVIATIONS,
        "training_config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": str(selected_device),
            "device_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "CPU",
        },
        "development": development,
        "selected_epoch": selected_epoch,
        "final_training_epochs": training_epochs,
        "final_epoch_rule": "development median best epoch" if final_epochs is None else "paper Fig. 6 curve length",
        "final": final,
        "summary": {
            "accuracy_mean": float(accuracies.mean()),
            "accuracy_std": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
            "balanced_accuracy_mean": float(balanced_accuracies.mean()),
            "balanced_accuracy_std": float(balanced_accuracies.std(ddof=1)) if len(balanced_accuracies) > 1 else 0.0,
            "macro_f1_mean": float(macro_f1.mean()),
            "macro_f1_std": float(macro_f1.std(ddof=1)) if len(macro_f1) > 1 else 0.0,
            "paper_claimed_accuracy": 0.9699,
            "paper_claimed_balanced_accuracy": 0.9699074074074074,
            "paper_claim_gap": float(accuracies.mean() - 0.9699),
            "paper_balanced_accuracy_gap": float(balanced_accuracies.mean() - 0.9699074074074074),
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "reproduction.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
