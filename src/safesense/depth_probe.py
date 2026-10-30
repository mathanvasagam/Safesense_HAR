"""Odd-subject-only depth complementarity ablation for the frozen SafeSense model.

This module deliberately does not add depth to SafeSense. It trains a compact depth-HOG
classifier on the four leave-one-odd-subject-out development folds and compares its
predictions with the preserved seed-22 SafeSenseFactorized SWA checkpoints. Even-subject
feature arrays are never opened by this workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.io import loadmat
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .preprocessing import DEPTH_FEATURES, FRAME_COUNT, depth_features, select_visual_indices
from .safesense import (
    SafeSenseDataset,
    SafeSenseError,
    SafeSenseFactorized,
    SafeSenseTrainingConfig,
    _development_data,
    _epoch_factorized,
    _loader,
    _metrics,
    _update_swa_state,
    parameter_count,
    runtime_environment,
    safesense_odd_fold_splits,
    set_safesense_seed,
)

ODD_SUBJECTS = (1, 3, 5, 7)
DEPTH_PROBE_SEEDS = (11, 22, 33)
DEPTH_PROBE_EPOCHS = 60
DEPTH_PROBE_SWA_START = 45
DEPTH_PROBE_HIDDEN = 128
DEPTH_PROBE_DROPOUT = 0.12


@dataclass(frozen=True)
class DepthProbeConfig:
    """Frozen training definition for the diagnostic depth branch."""

    epochs: int = DEPTH_PROBE_EPOCHS
    swa_start_epoch: int = DEPTH_PROBE_SWA_START
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = DEPTH_PROBE_DROPOUT
    hidden_size: int = DEPTH_PROBE_HIDDEN

    def __post_init__(self) -> None:
        if self.epochs < 1 or not 1 <= self.swa_start_epoch <= self.epochs:
            raise ValueError("invalid depth-probe SWA schedule")
        if self.batch_size < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid depth-probe optimizer configuration")
        if self.hidden_size < 1 or not 0 <= self.dropout < 1:
            raise ValueError("invalid depth-probe architecture configuration")


class DepthProbeDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Train-only-normalized HOG depth sequences."""

    def __init__(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        labels: np.ndarray,
        indices: Sequence[int],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        normalized = ((depth - mean) / std).astype(np.float32, copy=False)
        normalized = normalized * mask[..., None]
        self.depth = torch.from_numpy(normalized)
        self.mask = torch.from_numpy(mask.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, position: int) -> tuple[Tensor, Tensor, Tensor]:
        index = int(self.indices[position])
        return self.depth[index], self.mask[index], self.labels[index]


class DepthHOGProbe(nn.Module):
    """Small diagnostic classifier using the repository's existing depth-HOG features.

    The LSTM-128 width mirrors the already-audited Houssein-style depth branch. The
    classifier is intentionally minimal so this experiment measures whether the depth
    representation carries complementary class information rather than proposing a new
    production architecture.
    """

    def __init__(self, *, hidden_size: int = DEPTH_PROBE_HIDDEN, dropout: float = DEPTH_PROBE_DROPOUT) -> None:
        super().__init__()
        self.lstm = nn.LSTM(DEPTH_FEATURES, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 27)

    @staticmethod
    def _masked_mean(sequence: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1)
        return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, depth: Tensor, mask: Tensor) -> Tensor:
        if depth.ndim != 3 or depth.shape[1:] != (FRAME_COUNT, DEPTH_FEATURES):
            raise ValueError(f"depth must have shape [B,{FRAME_COUNT},{DEPTH_FEATURES}], got {tuple(depth.shape)}")
        if mask.shape != depth.shape[:2]:
            raise ValueError("depth mask must match batch/time dimensions")
        mask = mask.to(dtype=depth.dtype).clamp(0, 1)
        sequence, _ = self.lstm(depth * mask.unsqueeze(-1))
        pooled = self._masked_mean(sequence, mask)
        return self.classifier(self.dropout(self.norm(pooled)))


def depth_train_stats(depth: np.ndarray, mask: np.ndarray, train_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-HOG-feature z-score statistics using training frames only."""
    indices = np.asarray(train_indices, dtype=np.int64)
    values = depth[indices]
    valid = mask[indices].astype(bool)
    flattened = values[valid]
    if flattened.size == 0:
        raise SafeSenseError("depth probe training split has no valid depth frames")
    mean = flattened.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = flattened.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def paired_outcomes(reference: np.ndarray, probe: np.ndarray, targets: np.ndarray) -> dict[str, int | float]:
    """Return the paired correctness table and optimistic oracle complementarity."""
    reference = np.asarray(reference, dtype=np.int64)
    probe = np.asarray(probe, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if reference.shape != probe.shape or reference.shape != targets.shape:
        raise ValueError("paired predictions/targets must have identical shapes")
    reference_correct = reference == targets
    probe_correct = probe == targets
    both = int(np.sum(reference_correct & probe_correct))
    reference_only = int(np.sum(reference_correct & ~probe_correct))
    probe_only = int(np.sum(~reference_correct & probe_correct))
    both_wrong = int(np.sum(~reference_correct & ~probe_correct))
    total = int(len(targets))
    reference_errors = probe_only + both_wrong
    oracle_accuracy = (both + reference_only + probe_only) / max(total, 1)
    return {
        "total": total,
        "both_correct": both,
        "safesense_only_correct": reference_only,
        "depth_only_correct": probe_only,
        "both_wrong": both_wrong,
        "oracle_union_accuracy": float(oracle_accuracy),
        "oracle_gain_over_safesense_percentage_points": float(100.0 * probe_only / max(total, 1)),
        "depth_recovery_rate_of_safesense_errors": float(probe_only / max(reference_errors, 1)),
    }


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _extract_odd_depth(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract existing audited depth-HOG features while opening odd-subject MAT files only."""
    rows = _read_manifest(cache_dir / "manifest.csv")
    odd_rows = [row for row in rows if int(row["subject"]) in ODD_SUBJECTS]
    if len(odd_rows) != 431:
        raise SafeSenseError(f"expected 431 odd-subject samples, got {len(odd_rows)}")
    features = np.zeros((len(odd_rows), FRAME_COUNT, DEPTH_FEATURES), dtype=np.float32)
    masks = np.zeros((len(odd_rows), FRAME_COUNT), dtype=np.float32)
    sample_ids: list[str] = []
    subjects: list[int] = []
    labels: list[int] = []
    for index, row in enumerate(odd_rows):
        subject = int(row["subject"])
        if subject not in ODD_SUBJECTS:
            raise SafeSenseError("depth extraction attempted to open an even-subject sample")
        depth_path = cache_dir / row["depth"]
        skeleton_path = cache_dir / row["skeleton"]
        try:
            depth = np.asarray(loadmat(depth_path)["d_depth"])
            skeleton = np.asarray(loadmat(skeleton_path)["d_skel"])
        except (KeyError, OSError, ValueError) as error:
            raise SafeSenseError(f"cannot load odd depth sample {row['sample_id']}: {error}") from error
        selected = select_visual_indices(skeleton)
        features[index], masks[index] = depth_features(depth, selected)
        sample_ids.append(row["sample_id"])
        subjects.append(subject)
        labels.append(int(row["action"]) - 1)
        if (index + 1) % 50 == 0 or index + 1 == len(odd_rows):
            print(f"depth-extraction samples={index + 1}/{len(odd_rows)}", flush=True)
    return (
        features,
        masks,
        np.asarray(labels, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
        np.asarray(sample_ids),
    )


def _depth_loader(dataset: DepthProbeDataset, config: DepthProbeConfig, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def _depth_epoch(
    model: DepthHOGProbe,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for depth, mask, labels in loader:
            depth, mask, labels = depth.to(device), mask.to(device), labels.to(device)
            logits = model(depth, mask)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch
            total_samples += batch
            predictions.append(logits.detach().argmax(1).cpu().numpy())
            targets.append(labels.cpu().numpy())
    return total_loss / max(total_samples, 1), np.concatenate(predictions), np.concatenate(targets)


def _load_safesense_reference(
    prepared_dir: Path,
    checkpoint: Path,
    validation_subject: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    splits = safesense_odd_fold_splits(prepared_dir, validation_subject)
    skeleton, imu, mask, labels, stats = _development_data(prepared_dir, splits)
    valid = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_valid, stats)
    config = SafeSenseTrainingConfig(seed=22, epochs=60, patience=60, batch_size=32)
    model = SafeSenseFactorized().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    loss, predictions, targets = _epoch_factorized(model, _loader(valid, config, False, 22), device, None)

    fusion_rows = _read_manifest(prepared_dir / "manifest.csv")
    odd_rows = [row for row in fusion_rows if int(row["subject"]) in ODD_SUBJECTS]
    sample_ids = [odd_rows[int(position)]["sample_id"] for position in splits.development_valid]
    metrics = _metrics(predictions, targets)
    metrics["loss"] = float(loss)
    return predictions, targets, sample_ids, metrics


def _reference_accuracy_map(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[int, float] = {}
    for row in payload.get("development", {}).get("result_matrix", []):
        if int(row.get("seed", -1)) == 22:
            mapping[int(row["validation_subject"])] = float(row["clean_accuracy"])
    return mapping


def run_depth_complementarity_probe(
    cache_dir: str | Path,
    prepared_dir: str | Path,
    checkpoint_root: str | Path,
    output_path: str | Path,
    *,
    reference_results_path: str | Path | None = None,
    config: DepthProbeConfig | None = None,
    seeds: Sequence[int] = DEPTH_PROBE_SEEDS,
    device: str = "cpu",
) -> dict[str, object]:
    """Run the complete odd-fold depth ablation without opening official even-subject arrays."""
    started = time.perf_counter()
    cache_dir = Path(cache_dir)
    prepared_dir = Path(prepared_dir)
    checkpoint_root = Path(checkpoint_root)
    output_path = Path(output_path)
    config = config or DepthProbeConfig()
    seeds = tuple(int(seed) for seed in seeds)
    if seeds != DEPTH_PROBE_SEEDS:
        raise SafeSenseError(f"depth probe seeds are frozen as {DEPTH_PROBE_SEEDS}")
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")

    depth, depth_mask, labels, subjects, sample_ids = _extract_odd_depth(cache_dir)
    if not np.isin(subjects, ODD_SUBJECTS).all():
        raise SafeSenseError("depth probe contains non-odd subjects")
    reference_accuracy = _reference_accuracy_map(Path(reference_results_path) if reference_results_path else None)

    fold_results: list[dict[str, object]] = []
    pooled_safesense: list[np.ndarray] = []
    pooled_targets: list[np.ndarray] = []
    canonical_depth: list[np.ndarray] = []
    all_depth_seed_metrics: list[dict[str, object]] = []

    for validation_subject in ODD_SUBJECTS:
        train_indices = np.flatnonzero((subjects != validation_subject) & np.isin(subjects, ODD_SUBJECTS))
        valid_indices = np.flatnonzero(subjects == validation_subject)
        mean, std = depth_train_stats(depth, depth_mask, train_indices)
        train = DepthProbeDataset(depth, depth_mask, labels, train_indices, mean, std)
        valid = DepthProbeDataset(depth, depth_mask, labels, valid_indices, mean, std)

        checkpoint = checkpoint_root / f"fold-{validation_subject}.pt"
        if not checkpoint.is_file():
            raise SafeSenseError(f"missing preserved SafeSense checkpoint: {checkpoint}")
        safe_predictions, safe_targets, safe_ids, safe_metrics = _load_safesense_reference(
            prepared_dir, checkpoint, validation_subject, selected_device
        )
        depth_ids = sample_ids[valid_indices].tolist()
        if safe_ids != depth_ids:
            raise SafeSenseError(f"SafeSense/depth sample order mismatch for subject {validation_subject}")
        if not np.array_equal(safe_targets, labels[valid_indices]):
            raise SafeSenseError(f"SafeSense/depth target mismatch for subject {validation_subject}")
        expected_accuracy = reference_accuracy.get(validation_subject)
        replay_match = expected_accuracy is None or abs(float(safe_metrics["accuracy"]) - expected_accuracy) < 1e-12
        if not replay_match:
            raise SafeSenseError(
                f"SafeSense checkpoint replay mismatch on fold {validation_subject}: "
                f"observed={safe_metrics['accuracy']} expected={expected_accuracy}"
            )

        seed_records: list[dict[str, object]] = []
        canonical_predictions: np.ndarray | None = None
        for seed in seeds:
            seed_config = SafeSenseTrainingConfig(
                seed=seed,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                patience=config.epochs,
            )
            set_safesense_seed(seed)
            model = DepthHOGProbe(hidden_size=config.hidden_size, dropout=config.dropout).to(selected_device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
            swa_state: dict[str, Tensor] | None = None
            swa_count = 0
            for epoch in range(1, config.epochs + 1):
                train_loss, _, _ = _depth_epoch(
                    model,
                    _depth_loader(train, config, shuffle=True, seed=seed_config.seed + epoch),
                    selected_device,
                    optimizer,
                )
                if epoch >= config.swa_start_epoch:
                    swa_state, swa_count = _update_swa_state(model, swa_state, swa_count)
                if epoch == 1 or epoch == config.epochs or epoch % 10 == 0:
                    print(
                        f"depth-probe fold={validation_subject} seed={seed} epoch={epoch}/{config.epochs} "
                        f"train_loss={train_loss:.6f} swa_count={swa_count}",
                        flush=True,
                    )
            if swa_state is None:
                raise SafeSenseError("depth probe did not collect SWA states")
            model.load_state_dict(swa_state)
            valid_loss, depth_predictions, depth_targets = _depth_epoch(
                model,
                _depth_loader(valid, config, shuffle=False, seed=seed),
                selected_device,
                None,
            )
            if not np.array_equal(depth_targets, safe_targets):
                raise SafeSenseError("depth/SafeSense validation targets diverged")
            metrics = _metrics(depth_predictions, depth_targets)
            paired = paired_outcomes(safe_predictions, depth_predictions, depth_targets)
            seed_depth_only_ids = [
                sample_id
                for sample_id, safe_pred, depth_pred, target in zip(safe_ids, safe_predictions, depth_predictions, safe_targets)
                if safe_pred != target and depth_pred == target
            ]
            record = {
                "seed": seed,
                "validation_loss": float(valid_loss),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "swa_state_count": int(swa_count),
                "paired_with_safesense": paired,
                "depth_only_correct_sample_ids": seed_depth_only_ids,
            }
            seed_records.append(record)
            all_depth_seed_metrics.append({"validation_subject": validation_subject, **record})
            if seed == 22:
                canonical_predictions = depth_predictions.copy()
            print(
                f"depth-probe fold={validation_subject} seed={seed} evaluated "
                f"accuracy={record['accuracy']:.6f} depth_only_fixes={paired['depth_only_correct']}",
                flush=True,
            )

        if canonical_predictions is None:
            raise SafeSenseError("canonical depth seed 22 was not evaluated")
        canonical_pair = paired_outcomes(safe_predictions, canonical_predictions, safe_targets)
        depth_only_ids = [
            sample_id
            for sample_id, safe_pred, depth_pred, target in zip(safe_ids, safe_predictions, canonical_predictions, safe_targets)
            if safe_pred != target and depth_pred == target
        ]
        safesense_only_ids = [
            sample_id
            for sample_id, safe_pred, depth_pred, target in zip(safe_ids, safe_predictions, canonical_predictions, safe_targets)
            if safe_pred == target and depth_pred != target
        ]
        seed_fix_sets = [set(record["depth_only_correct_sample_ids"]) for record in seed_records]
        stable_all_seed_ids = sorted(set.intersection(*seed_fix_sets)) if seed_fix_sets else []
        fix_frequency = {
            sample_id: sum(sample_id in fixes for fixes in seed_fix_sets)
            for sample_id in sorted(set.union(*seed_fix_sets)) if seed_fix_sets
        }
        majority_seed_ids = sorted(sample_id for sample_id, count in fix_frequency.items() if count >= 2)
        any_seed_ids = sorted(fix_frequency)
        fold_results.append(
            {
                "validation_subject": validation_subject,
                "training_subjects": [subject for subject in ODD_SUBJECTS if subject != validation_subject],
                "training_count": int(len(train_indices)),
                "validation_count": int(len(valid_indices)),
                "safesense_seed22": {
                    "accuracy": float(safe_metrics["accuracy"]),
                    "macro_f1": float(safe_metrics["macro_f1"]),
                    "recorded_accuracy": expected_accuracy,
                    "checkpoint_replay_exact_accuracy_match": bool(replay_match),
                },
                "depth_seed_results": seed_records,
                "canonical_seed22_paired": canonical_pair,
                "depth_only_correct_sample_ids": depth_only_ids,
                "safesense_only_correct_sample_ids": safesense_only_ids,
                "depth_rescue_stability": {
                    "all_three_seeds_count": len(stable_all_seed_ids),
                    "all_three_seeds_sample_ids": stable_all_seed_ids,
                    "at_least_two_seeds_count": len(majority_seed_ids),
                    "at_least_two_seeds_sample_ids": majority_seed_ids,
                    "any_seed_count": len(any_seed_ids),
                    "any_seed_sample_ids": any_seed_ids,
                    "per_sample_seed_frequency": fix_frequency,
                },
            }
        )
        pooled_safesense.append(safe_predictions)
        pooled_targets.append(safe_targets)
        canonical_depth.append(canonical_predictions)

    safe_all = np.concatenate(pooled_safesense)
    targets_all = np.concatenate(pooled_targets)
    depth_all = np.concatenate(canonical_depth)
    aggregate_pair = paired_outcomes(safe_all, depth_all, targets_all)
    safe_metrics = _metrics(safe_all, targets_all)
    depth_metrics = _metrics(depth_all, targets_all)
    depth_seed_accuracies = [float(row["accuracy"]) for row in all_depth_seed_metrics]
    depth_seed_f1 = [float(row["macro_f1"]) for row in all_depth_seed_metrics]
    stable_all_seed_ids = sorted(
        sample_id
        for fold in fold_results
        for sample_id in fold["depth_rescue_stability"]["all_three_seeds_sample_ids"]
    )
    majority_seed_ids = sorted(
        sample_id
        for fold in fold_results
        for sample_id in fold["depth_rescue_stability"]["at_least_two_seeds_sample_ids"]
    )
    any_seed_ids = sorted(
        sample_id
        for fold in fold_results
        for sample_id in fold["depth_rescue_stability"]["any_seed_sample_ids"]
    )
    canonical_folds_with_rescues = [
        int(fold["validation_subject"])
        for fold in fold_results
        if int(fold["canonical_seed22_paired"]["depth_only_correct"]) > 0
    ]

    oracle_gain = float(aggregate_pair["oracle_gain_over_safesense_percentage_points"])
    recovery = float(aggregate_pair["depth_recovery_rate_of_safesense_errors"])
    meaningful = oracle_gain >= 1.0 and recovery >= 0.20
    conclusion = (
        "meaningful complementarity signal; depth is eligible for a separate future development-only privileged/optional-modality experiment"
        if meaningful
        else "insufficient complementarity to justify a depth integration cycle; retain skeleton+IMU as the selected model"
    )

    result: dict[str, object] = {
        "date": "2026-08-16",
        "model": "SafeSense",
        "workflow": "odd_subject_depth_hog_complementarity_ablation",
        "status": "completed",
        "purpose": "diagnostic ablation only; depth is not added to the selected SafeSense model",
        "protocol": {
            "development_subjects": list(ODD_SUBJECTS),
            "leave_one_odd_subject_out": True,
            "official_even_subject_arrays_loaded": False,
            "official_even_subjects_used_for_selection": False,
            "safesense_reference": "preserved SafeSenseFactorized seed-22 late-SWA odd-fold checkpoints",
            "depth_seeds": list(seeds),
            "canonical_paired_depth_seed": 22,
            "depth_training": asdict(config),
            "depth_preprocessing": (
                "existing audited 32-frame 92-D HOG depth pipeline; frame indices are selected by odd-subject skeleton motion ranking"
            ),
            "depth_probe_interpretation_caveat": (
                "the classifier input is depth HOG, but temporal frame selection uses skeleton motion; this is an intentionally favorable complementarity probe, not a strict depth-only deployment pipeline"
            ),
            "meaningful_complementarity_rule_preregistered_before_results": (
                "canonical seed-22 oracle union gain >= 1.0 percentage point AND depth recovers >= 20% of SafeSense errors"
            ),
        },
        "depth_model": {
            "architecture": "LSTM(92->128) + masked temporal mean + LayerNorm + dropout(0.12) + Linear(128->27)",
            "parameter_count": parameter_count(DepthHOGProbe()),
            "role": "diagnostic classifier only; not a proposed SafeSense production branch",
        },
        "folds": fold_results,
        "aggregate": {
            "sample_count": int(len(targets_all)),
            "safesense_seed22_pooled_accuracy": float(safe_metrics["accuracy"]),
            "safesense_seed22_pooled_macro_f1": float(safe_metrics["macro_f1"]),
            "depth_seed22_pooled_accuracy": float(depth_metrics["accuracy"]),
            "depth_seed22_pooled_macro_f1": float(depth_metrics["macro_f1"]),
            "depth_all_seed_fold_accuracy_mean": float(np.mean(depth_seed_accuracies)),
            "depth_all_seed_fold_accuracy_std_population": float(np.std(depth_seed_accuracies)),
            "depth_all_seed_fold_macro_f1_mean": float(np.mean(depth_seed_f1)),
            "canonical_seed22_paired": aggregate_pair,
            "depth_rescue_stability_across_seeds": {
                "all_three_seeds_count": len(stable_all_seed_ids),
                "all_three_seeds_sample_ids": stable_all_seed_ids,
                "at_least_two_seeds_count": len(majority_seed_ids),
                "at_least_two_seeds_sample_ids": majority_seed_ids,
                "any_seed_count": len(any_seed_ids),
                "any_seed_sample_ids": any_seed_ids,
                "canonical_seed22_folds_with_rescues": canonical_folds_with_rescues,
            },
        },
        "decision": {
            "meaningful_complementarity_gate_passed": bool(meaningful),
            "preregistered_gate_conclusion": conclusion,
            "posthoc_stability_interpretation": (
                "the canonical gate is a screening result only; rescue stability and fold concentration must be reported, and this ablation alone does not authorize depth integration"
            ),
            "recommended_use": "paper ablation evidence only; retain the frozen skeleton+IMU SafeSense model",
            "no_even_subject_evaluation_authorized": True,
            "selected_safesense_model_changed": False,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": runtime_environment(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-results")
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_depth_complementarity_probe(
        args.cache_dir,
        args.prepared_dir,
        args.checkpoint_root,
        args.output,
        reference_results_path=args.reference_results,
        device=args.device,
    )
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"], "decision": result["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
