"""Command-line entry point for the experiment stages."""

import argparse
import json
from pathlib import Path

from .dataset import DatasetError, prepare_dataset
from .fusion_gcn import (
    FusionGCNError,
    prepare_fusion_gcn_dataset,
    runtime_reproduce,
    runtime_smoke,
)
from .safesense import SAFESENSE_VARIANTS, SafeSenseTrainingConfig, SafeSenseError, evaluate_safesense, safesense_smoke, run_safesense_development, run_safesense_final
from .preprocessing import PreprocessingError, preprocess_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safesense")
    commands = parser.add_subparsers(dest="command")
    prepare = commands.add_parser("prepare", help="extract, index, and preprocess UTD-MHAD")
    prepare.add_argument("--raw-dir", required=True, help="directory containing the UTD-MHAD ZIP archives")
    prepare.add_argument(
        "--cache-dir",
        required=True,
        help="directory for extracted files, manifest, feature cache, and preprocessing metadata",
    )
    fusion_prepare = commands.add_parser(
        "prepare-fusion", aliases=("fusion-prepare",), help="prepare the skeleton/IMU Fusion-GCN artifact"
    )
    fusion_prepare.add_argument("--data-dir", required=True, help="directory containing Skeleton and Inertial MAT files")
    fusion_prepare.add_argument("--output-dir", required=True, help="directory for official-compatible arrays and audit")
    fusion_prepare.add_argument("--expected-count", type=int, default=861)
    for name, help_text in (
        ("fusion-smoke", "run the one-epoch Fusion-GCN runtime smoke"),
        ("fusion-reproduce", "run the final-epoch Fusion-GCN reproduction"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--repo-dir", required=True, help="external runtime checkout of mduhme/fusion-gcn")
        command.add_argument("--prepared-dir", required=True, help="output of prepare-fusion")
        command.add_argument("--output-dir", required=True, help="directory for runtime logs and compact JSON")
        command.add_argument("--seed", type=int, default=1)
    train = commands.add_parser("train", help="train SafeSense")
    train.add_argument("--prepared-dir", help="prepared cache/fusion-gcn directory")
    train.add_argument("--output-dir", help="directory for SafeSense artifacts")
    train.add_argument("--device", default="cpu")
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--variant", choices=SAFESENSE_VARIANTS, default="full")
    train.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    final = commands.add_parser("final", help="train/evaluate SafeSense with a frozen development epoch")
    final.add_argument("--prepared-dir", required=True)
    final.add_argument("--output-dir", required=True)
    final.add_argument("--frozen-epoch", type=int, required=True)
    final.add_argument("--device", default="cpu")
    final.add_argument("--batch-size", type=int, default=32)
    final.add_argument("--learning-rate", type=float, default=1e-3)
    final.add_argument("--weight-decay", type=float, default=1e-4)
    final.add_argument("--patience", type=int, default=20)
    final.add_argument("--variant", choices=SAFESENSE_VARIANTS, default="full")
    final.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    smoke = commands.add_parser("smoke", help="run a one-epoch SafeSense smoke")
    smoke.add_argument("--prepared-dir", required=True)
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--variant", choices=SAFESENSE_VARIANTS, default="full")
    smoke.add_argument("--device", default="cpu")
    evaluate = commands.add_parser("evaluate", help="evaluate a completed SafeSense run")
    evaluate.add_argument("--prepared-dir")
    evaluate.add_argument("--run-dir")
    evaluate.add_argument("--device", default="cpu")
    report = commands.add_parser("report", help="report a completed SafeSense run")
    report.add_argument("--run-dir")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command is None:
        build_parser().print_help()
        return 0
    if args.command == "prepare":
        try:
            index_path = prepare_dataset(args.raw_dir, args.cache_dir)
            features_path = preprocess_dataset(args.cache_dir)
        except (DatasetError, PreprocessingError) as error:
            print(f"safesense prepare: {error}")
            return 2
        print(
            f"indexed 861 UTD-MHAD triplets: {index_path} "
            f"(manifest: {index_path.parent / 'manifest.csv'}; features: {features_path})"
        )
        return 0
    if args.command in {"prepare-fusion", "fusion-prepare"}:
        try:
            audit_path = prepare_fusion_gcn_dataset(
                args.data_dir,
                args.output_dir,
                expected_count=args.expected_count,
            )
        except FusionGCNError as error:
            print(f"safesense prepare-fusion: {error}")
            return 2
        print(json.dumps({"status": "prepared", "audit": str(audit_path)}, sort_keys=True))
        return 0
    if args.command in {"fusion-smoke", "fusion-reproduce"}:
        try:
            result = (runtime_smoke if args.command == "fusion-smoke" else runtime_reproduce)(
                args.repo_dir,
                args.prepared_dir,
                args.output_dir,
                seed=args.seed,
            )
        except FusionGCNError as error:
            print(f"safesense {args.command}: {error}")
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("status") == "completed" else 2
    if args.command == "train":
        if not args.prepared_dir or not args.output_dir:
            print("safesense train: --prepared-dir and --output-dir are required")
            return 2
        try:
            result = run_safesense_development(
                args.prepared_dir,
                args.output_dir,
                config=SafeSenseTrainingConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                ),
                seeds=args.seeds,
                variant=args.variant,
                device=args.device,
            )
        except SafeSenseError as error:
            print(f"safesense train: {error}")
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "final":
        try:
            result = run_safesense_final(
                args.prepared_dir,
                args.output_dir,
                frozen_epoch=args.frozen_epoch,
                config=SafeSenseTrainingConfig(
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                ),
                seeds=args.seeds,
                variant=args.variant,
                device=args.device,
            )
        except SafeSenseError as error:
            print(f"safesense final: {error}")
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "smoke":
        try:
            result = safesense_smoke(args.prepared_dir, args.output_dir, variant=args.variant, device=args.device)
        except SafeSenseError as error:
            print(f"safesense smoke: {error}")
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "evaluate":
        if not args.prepared_dir or not args.run_dir:
            print("safesense evaluate: --prepared-dir and --run-dir are required")
            return 2
        try:
            result = evaluate_safesense(args.prepared_dir, args.run_dir, device=args.device)
        except SafeSenseError as error:
            print(f"safesense evaluate: {error}")
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "report":
        if not args.run_dir:
            print("safesense report: --run-dir is required")
            return 2
        result_path = Path(args.run_dir) / "result.json"
        if not result_path.is_file():
            print(f"safesense report: missing result: {result_path}")
            return 2
        print(result_path.read_text(encoding="utf-8"), end="")
        return 0
    print(f"safesense {args.command}: implementation is scheduled for a later Day 1 task")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
