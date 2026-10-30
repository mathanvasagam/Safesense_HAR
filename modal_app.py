"""Modal entry points for environment checks and baseline runs."""

from time import perf_counter

import modal


app = modal.App("safesense")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("matplotlib", "tensorboard", "PyYAML", "networkx", "pandas", "seaborn")
    .run_commands("git clone --depth 1 https://github.com/mduhme/fusion-gcn /root/fusion-gcn")
    .add_local_file("src/safesense/__init__.py", remote_path="/root/safesense/__init__.py")
    .add_local_file("src/safesense/config.py", remote_path="/root/safesense/config.py")
    .add_local_file("src/safesense/model.py", remote_path="/root/safesense/model.py")
    .add_local_file("src/safesense/baseline.py", remote_path="/root/safesense/baseline.py")
    .add_local_file("src/safesense/fusion_gcn.py", remote_path="/root/safesense/fusion_gcn.py")
    .add_local_file("src/safesense/safesense.py", remote_path="/root/safesense/safesense.py")
    .add_local_file("src/safesense/robustness.py", remote_path="/root/safesense/robustness.py")
    .add_local_file("src/safesense/efficiency.py", remote_path="/root/safesense/efficiency.py")
    .add_local_file("scripts/run_matched_comparison_baselines.py", remote_path="/root/run_matched_comparison_baselines.py")
    .add_local_dir("cache/fusion-gcn", remote_path="/root/fusion-prepared")
    .add_local_file("cache/features.npz", remote_path="/root/features.npz")
    .add_local_file("cache/fusion-gcn-seed11-final.pt", remote_path="/root/fusion-gcn-seed11-final.pt")
)
results_volume = modal.Volume.from_name("safesense-baseline-results", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=60 * 60 * 4, volumes={"/results": results_volume})
def matched_comparison_baselines() -> dict[str, object]:
    """Run standard unimodal/early/late fusion baselines on the frozen 25-scenario suite."""
    import json
    import subprocess
    from pathlib import Path

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("matched comparison baselines require CUDA")
    output = Path("/results/matched-standard-baselines")
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.log"
    command = [
        "python",
        "/root/run_matched_comparison_baselines.py",
        "--prepared",
        "/root/fusion-prepared",
        "--output",
        str(output / "result.json"),
        "--predictions",
        str(output / "predictions.json"),
        "--device",
        "cuda",
    ]
    with progress.open("a", encoding="utf-8") as stream:
        stream.write("starting matched comparison baselines\n")
        stream.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        return_code = process.wait()
        stream.write(f"return_code={return_code}\n")
        stream.flush()
    if return_code:
        results_volume.commit()
        raise RuntimeError(f"matched comparison baseline process failed with return code {return_code}")
    results_volume.commit()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    return {
        "device": torch.cuda.get_device_name(0),
        "result_path": str(output / "result.json"),
        "predictions_path": str(output / "predictions.json"),
        "progress_path": str(progress),
        "models": {
            name: {
                "clean_accuracy_mean": values["clean_accuracy_mean"],
                "equal_category_corrupted_accuracy_mean": values["equal_category_corrupted_accuracy_mean"],
            }
            for name, values in result["models"].items()
        },
    }


@app.function(image=image, gpu="T4")
def environment_smoke() -> dict[str, str]:
    """Confirm that the Modal T4 image can access CUDA and PyTorch."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this function requires a T4")
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda or "unknown",
        "device": torch.cuda.get_device_name(0),
    }


@app.function(image=image, gpu="T4")
def model_smoke() -> dict[str, str | int | float]:
    """Run one deterministic HousseinBaseline optimization step on a T4."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; model smoke requires a T4")
    device_name = torch.cuda.get_device_name(0)
    if "t4" not in device_name.lower():
        raise RuntimeError(f"model smoke requires a T4, got {device_name!r}")

    from safesense.model import HousseinBaseline

    torch.manual_seed(1234)
    device = torch.device("cuda")
    model = HousseinBaseline().to(device)
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=9.8e-4, weight_decay=1.6e-4)
    depth = torch.arange(4 * 32 * 92, device=device, dtype=torch.float32).reshape(4, 32, 92) / 1000
    skeleton = torch.arange(4 * 32 * 60, device=device, dtype=torch.float32).reshape(4, 32, 60) / 1000
    imu = torch.arange(4 * 32 * 6, device=device, dtype=torch.float32).reshape(4, 32, 6) / 1000
    labels = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)

    torch.cuda.synchronize()
    started = perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits, weights = model(depth, skeleton, imu)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed_ms = (perf_counter() - started) * 1000

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if logits.shape != (4, 27) or weights.shape != (4, 3):
        raise RuntimeError(f"unexpected output shapes: logits={tuple(logits.shape)}, weights={tuple(weights.shape)}")
    if not all(torch.isfinite(value).all() for value in (logits, weights, loss, *gradients)):
        raise RuntimeError("non-finite model smoke output, loss, or gradient")
    if not torch.allclose(weights.sum(dim=1), torch.ones(4, device=device), atol=1e-5):
        raise RuntimeError("modality weights do not sum to one")

    result: dict[str, str | int | float] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda or "unknown",
        "device": device_name,
        "loss": float(loss.detach().cpu()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_ms": elapsed_ms,
    }
    print(result)
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60, volumes={"/results": results_volume})
def baseline_smoke() -> dict[str, str | int | float]:
    """Run one development and one fixed-epoch test pass on the Modal T4."""
    import torch

    from safesense.baseline import TrainingConfig, run_reproduction

    result = run_reproduction(
        "/root/features.npz",
        "/results/smoke",
        development_seeds=(11,),
        final_seeds=(11,),
        training_config=TrainingConfig(max_epochs=1),
        device="cuda",
    )
    results_volume.commit()
    summary = result["summary"]
    return {
        "device": torch.cuda.get_device_name(0),
        "selected_epoch": result["selected_epoch"],
        "accuracy": summary["accuracy_mean"],
        "balanced_accuracy": summary["balanced_accuracy_mean"],
        "macro_f1": summary["macro_f1_mean"],
    }


@app.function(image=image, gpu="T4", timeout=60 * 60 * 4, volumes={"/results": results_volume})
def baseline_reproduction(variant: str = "table2") -> dict[str, str | int | float]:
    """Run the frozen protocol with either of the paper's dropout disclosures."""
    import torch

    from safesense.baseline import TrainingConfig, run_reproduction

    variants = {
        "table2": (TrainingConfig(), "/results/clean-table2-v2", None),
        "narrative": (
            TrainingConfig(branch_dropout=0.3, classifier_dropout=0.5),
            "/results/narrative-dropout",
            None,
        ),
        "paper-curve": (
            TrainingConfig(branch_dropout=0.3, classifier_dropout=0.5),
            "/results/paper-curve-105",
            105,
        ),
        "parameter-informed": (
            TrainingConfig(classifier_width=1024),
            "/results/clean-parameter-informed-v2",
            None,
        ),
    }
    if variant not in variants:
        raise ValueError(f"variant must be one of {tuple(variants)}, got {variant!r}")
    training_config, output_dir, final_epochs = variants[variant]
    result = run_reproduction(
        "/root/features.npz",
        output_dir,
        training_config=training_config,
        final_epochs=final_epochs,
        device="cuda",
    )
    results_volume.commit()
    summary = result["summary"]
    return {
        "variant": variant,
        "device": torch.cuda.get_device_name(0),
        "selected_epoch": result["selected_epoch"],
        "final_training_epochs": result["final_training_epochs"],
        "accuracy_mean": summary["accuracy_mean"],
        "accuracy_std": summary["accuracy_std"],
        "balanced_accuracy_mean": summary["balanced_accuracy_mean"],
        "balanced_accuracy_std": summary["balanced_accuracy_std"],
        "macro_f1_mean": summary["macro_f1_mean"],
        "macro_f1_std": summary["macro_f1_std"],
        "paper_claimed_accuracy": summary["paper_claimed_accuracy"],
    }


@app.function(image=image, gpu="T4", timeout=60 * 60 * 2, volumes={"/results": results_volume})
def paper_visible_diagnostic(variant: str = "table2") -> dict[str, str | int | float]:
    """Run explicitly leaky odd/even diagnostic variants on a T4."""
    import torch

    from safesense.baseline import TrainingConfig, run_paper_visible_diagnostic

    variants = {
        "table2": (0.12, 0.82, 128, "/results/diagnostic-table2"),
        "narrative": (0.3, 0.5, 128, "/results/diagnostic-narrative"),
        "parameter-informed": (0.12, 0.82, 1024, "/results/diagnostic-parameter-informed"),
    }
    if variant not in variants:
        raise ValueError(f"variant must be one of {tuple(variants)}, got {variant!r}")
    branch_dropout, classifier_dropout, classifier_width, output_dir = variants[variant]
    result = run_paper_visible_diagnostic(
        "/root/features.npz",
        output_dir,
        seeds=(11, 22, 33),
        training_config=TrainingConfig(
            max_epochs=105,
            branch_dropout=branch_dropout,
            classifier_dropout=classifier_dropout,
            classifier_width=classifier_width,
        ),
        device="cuda",
    )
    results_volume.commit()
    accuracies = [float(run["accuracy"]) for run in result["runs"]]
    return {
        "variant": variant,
        "diagnostic_status": result["diagnostic_status"],
        "device": torch.cuda.get_device_name(0),
        "classifier_width": classifier_width,
        "accuracy_mean": sum(accuracies) / len(accuracies),
    }


@app.function(image=image, gpu="T4", timeout=60 * 60, volumes={"/results": results_volume})
def fusion_smoke(
    repo_dir: str = "/root/fusion-gcn",
    prepared_dir: str = "/root/fusion-prepared",
    output_dir: str = "/results/fusion-smoke",
) -> dict[str, object]:
    """Run a one-epoch center-node Fusion-GCN smoke from an external checkout."""
    from safesense.fusion_gcn import runtime_smoke

    result = runtime_smoke(repo_dir, prepared_dir, output_dir)
    results_volume.commit()
    if result.get("status") != "completed":
        raise RuntimeError(f"Fusion-GCN smoke failed: {result.get('error') or result.get('stderr_tail', '')[-500:]}")
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 4, volumes={"/results": results_volume})
def fusion_reproduction(
    repo_dir: str = "/root/fusion-gcn",
    prepared_dir: str = "/root/fusion-prepared",
    output_dir: str = "/results/fusion-reproduction",
) -> dict[str, object]:
    """Run the official 60-epoch center-node Fusion-GCN configuration."""
    from safesense.fusion_gcn import runtime_reproduce

    result = runtime_reproduce(repo_dir, prepared_dir, output_dir)
    results_volume.commit()
    if result.get("status") != "completed":
        raise RuntimeError(f"Fusion-GCN reproduction failed: {result.get('error') or result.get('stderr_tail', '')[-500:]}")
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 4, volumes={"/results": results_volume})
def fusion_paired_seed(seed: int) -> dict[str, object]:
    """Run one official Fusion-GCN paired final seed in an isolated result directory."""
    if seed not in (11, 22, 33):
        raise ValueError("Fusion-GCN paired seed must be one of 11, 22, 33")
    from safesense.fusion_gcn import runtime_reproduce

    output_dir = f"/results/fusion-paired/seed-{seed}"
    result = runtime_reproduce("/root/fusion-gcn", "/root/fusion-prepared", output_dir, seed=seed)
    results_volume.commit()
    if result.get("status") != "completed":
        raise RuntimeError(f"Fusion-GCN paired seed failed: {result.get('error') or result.get('stderr_tail', '')[-500:]}")
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60, volumes={"/results": results_volume})
def safesense_smoke(
    prepared_dir: str = "/root/fusion-prepared",
    output_dir: str = "/results/safesense-smoke",
    variant: str = "full",
) -> dict[str, object]:
    """Run one deterministic SafeSense epoch on the prepared skeleton/IMU cache."""
    from safesense.safesense import safesense_smoke as run_smoke

    result = run_smoke(prepared_dir, output_dir, variant=variant, device="cuda")
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 4, volumes={"/results": results_volume})
def safesense_development(
    prepared_dir: str = "/root/fusion-prepared",
    output_dir: str = "/results/safesense-development",
) -> dict[str, object]:
    """Select a frozen epoch on subjects 1,3,5 -> 7 using seeds 11,22,33."""
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_development

    result = run_safesense_development(
        prepared_dir,
        output_dir,
        config=SafeSenseTrainingConfig(epochs=120, patience=20),
        seeds=(11, 22, 33),
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 4, volumes={"/results": results_volume})
def safesense_final(
    frozen_epoch: int = 15,
    variant: str = "full",
    prepared_dir: str = "/root/fusion-prepared",
    output_dir: str = "/results/safesense-final",
) -> dict[str, object]:
    """Run a selected SafeSense variant at a supplied frozen epoch."""
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_final

    result = run_safesense_final(
        prepared_dir,
        output_dir,
        frozen_epoch=frozen_epoch,
        config=SafeSenseTrainingConfig(epochs=frozen_epoch, patience=20),
        seeds=(11, 22, 33),
        variant=variant,
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 6, volumes={"/results": results_volume})
def safesense_ablations(
    prepared_dir: str = "/root/fusion-prepared",
    output_dir: str = "/results/safesense-ablations",
) -> dict[str, object]:
    """Run frozen equal-fusion and single-modality SafeSense ablations."""
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_final

    results: dict[str, object] = {}
    for variant in ("equal", "skeleton", "imu"):
        results[variant] = run_safesense_final(
            prepared_dir,
            f"{output_dir}/{variant}",
            frozen_epoch=15,
            config=SafeSenseTrainingConfig(epochs=15, patience=20),
            seeds=(11, 22, 33),
            variant=variant,
            device="cuda",
        )
    results_volume.commit()
    return results


@app.function(image=image, gpu="T4", timeout=60 * 60 * 6, volumes={"/results": results_volume})
def safesense_development_candidates(
    prepared_dir: str = "/root/fusion-prepared",
    output_dir: str = "/results/safesense-development-candidates",
) -> dict[str, object]:
    """Compare equal fusion and classwise routing on the development split."""
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_development

    results: dict[str, object] = {}
    for variant in ("equal", "classwise"):
        results[variant] = run_safesense_development(
            prepared_dir,
            f"{output_dir}/{variant}",
            config=SafeSenseTrainingConfig(epochs=120, patience=20),
            seeds=(11, 22, 33),
            variant=variant,
            device="cuda",
        )
    results_volume.commit()
    return results


@app.function(image=image, gpu="T4", timeout=60 * 60 * 6, volumes={"/results": results_volume})
def safesense_classprior_development(
    prepared_dir: str = "/root/fusion-prepared",
    output_dir: str = "/results/safesense-classprior-development",
) -> dict[str, object]:
    """Evaluate the stable class-prior candidate on the development split only."""
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_development

    result = run_safesense_development(
        prepared_dir,
        output_dir,
        config=SafeSenseTrainingConfig(epochs=120, patience=20),
        seeds=(11, 22, 33),
        variant="classprior",
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 8, volumes={"/results": results_volume})
def safesense_alignment_development(
    prepared_dir: str = "/root/fusion-prepared",
    output_dir: str = "/results/safesense-alignment-development",
) -> dict[str, object]:
    """Compare classprior and aligned SafeSense using only subjects 1,3,5 -> 7."""
    import json
    from pathlib import Path

    from safesense.robustness import run_safesense_development_robustness
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_development

    summary: dict[str, object] = {
        "model": "SafeSense",
        "workflow": "alignment_development_comparison",
        "development_train_subjects": [1, 3, 5],
        "development_validation_subjects": [7],
        "even_subjects_used": False,
        "seeds": [11, 22, 33],
        "variants": {},
    }
    config = SafeSenseTrainingConfig(epochs=120, patience=20)
    for variant in ("classprior", "aligned"):
        variant_dir = f"{output_dir}/{variant}"
        development = run_safesense_development(
            prepared_dir,
            variant_dir,
            config=config,
            seeds=(11, 22, 33),
            variant=variant,
            device="cuda",
        )
        robustness = run_safesense_development_robustness(
            prepared_dir,
            variant_dir,
            f"{variant_dir}/robustness",
            device="cuda",
        )
        seed_results = development["seed_results"]
        summary["variants"][variant] = {
            "parameter_count": development["parameter_count"],
            "frozen_epoch": development["frozen_epoch"],
            "clean_accuracy_mean": sum(float(item["best_validation_accuracy"]) for item in seed_results) / len(seed_results),
            "clean_macro_f1_mean": sum(float(item["best_validation_macro_f1"]) for item in seed_results) / len(seed_results),
            "robustness": robustness["model_aggregate"],
            "lag_recovery": robustness["lag_recovery"],
        }
    path = Path(output_dir) / "comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    return summary


@app.function(image=image, gpu="T4", timeout=60 * 60 * 8, volumes={"/results": results_volume})
def safesense_temporal_development(
    prepared_dir: str = "/root/fusion-prepared",
    output_dir: str = "/results/safesense-temporal-balanced-development",
) -> dict[str, object]:
    """Compare selected temporal SafeSense with classprior on subjects 1,3,5 -> 7."""
    import json
    from pathlib import Path

    from safesense.robustness import run_safesense_development_robustness
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_development

    summary: dict[str, object] = {
        "model": "SafeSense",
        "workflow": "temporal_balanced_development_comparison",
        "development_train_subjects": [1, 3, 5],
        "development_validation_subjects": [7],
        "even_subjects_used": False,
        "seeds": [11, 22, 33],
        "variants": {},
    }
    config = SafeSenseTrainingConfig(epochs=120, patience=20)
    for variant in ("classprior", "temporal"):
        variant_dir = f"{output_dir}/{variant}"
        development = run_safesense_development(
            prepared_dir,
            variant_dir,
            config=config,
            seeds=(11, 22, 33),
            variant=variant,
            device="cuda",
        )
        robustness = run_safesense_development_robustness(
            prepared_dir,
            variant_dir,
            f"{variant_dir}/robustness",
            device="cuda",
        )
        seed_results = development["seed_results"]
        summary["variants"][variant] = {
            "parameter_count": development["parameter_count"],
            "frozen_epoch": development["frozen_epoch"],
            "clean_accuracy_mean": sum(float(item["best_validation_accuracy"]) for item in seed_results) / len(seed_results),
            "clean_macro_f1_mean": sum(float(item["best_validation_macro_f1"]) for item in seed_results) / len(seed_results),
            "robustness": robustness["model_aggregate"],
            "lag_recovery": robustness["lag_recovery"],
            "consistency_weight": development.get("consistency_weight", 0.0),
            "clean_view_weight": development.get("clean_view_weight", 1.0),
        }
    path = Path(output_dir) / "comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    return summary


@app.function(image=image, gpu="T4", timeout=60 * 60 * 2, volumes={"/results": results_volume})
def safesense_temporal_fixed33_development_seed(
    seed: int,
    prepared_dir: str = "/root/fusion-prepared",
    output_root: str = "/results/safesense-temporal-fixed33-selected-development-split",
) -> dict[str, object]:
    """Train one subject-1/3/5 temporal base for the pre-selected 33 epochs, then evaluate subject 7."""
    if seed not in (11, 22, 33):
        raise ValueError("seed must be one of 11, 22, 33")
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_development_fixed_epoch

    result = run_safesense_development_fixed_epoch(
        prepared_dir,
        f"{output_root}/seedrun-{seed}",
        frozen_epoch=33,
        config=SafeSenseTrainingConfig(epochs=33, patience=20),
        seeds=(seed,),
        variant="temporal",
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 2, volumes={"/results": results_volume})
def safesense_temporal_fixed33_development_consolidate(
    prepared_dir: str = "/root/fusion-prepared",
    split_root: str = "/results/safesense-temporal-fixed33-selected-development-split",
    output_dir: str = "/results/safesense-temporal-fixed33-selected-development-complete",
) -> dict[str, object]:
    """Consolidate fixed-epoch temporal development bases and run subject-7 robustness."""
    import json
    import shutil
    from pathlib import Path

    from safesense.robustness import run_safesense_development_robustness
    from safesense.safesense import SafeSense, SafeSenseTrainingConfig, parameter_count, runtime_environment

    records = []
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for seed in (11, 22, 33):
        run_dir = Path(split_root) / f"seedrun-{seed}"
        recorded = json.loads((run_dir / "development.json").read_text(encoding="utf-8"))
        if recorded.get("training_mode") != "fixed_epoch_development_control" or recorded.get("seeds") != [seed]:
            raise RuntimeError(f"unexpected fixed temporal seed artifact for {seed}")
        record = recorded["seed_results"][0]
        if int(record["best_epoch"]) != 33:
            raise RuntimeError(f"fixed temporal seed {seed} did not use epoch 33")
        records.append(record)
        source = run_dir / f"seed-{seed}"
        destination = output / f"seed-{seed}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    development = {
        "model": "SafeSense",
        "workflow": "development",
        "status": "completed",
        "variant": "temporal",
        "training_mode": "fixed_epoch_development_control",
        "seeds": [11, 22, 33],
        "best_epochs": [33, 33, 33],
        "frozen_epoch": 33,
        "epoch_rule": "shared epoch selected by mean subject-7 validation accuracy across seeds",
        "config": SafeSenseTrainingConfig(epochs=33, patience=20).to_dict(),
        "development_train_subjects": [1, 3, 5],
        "development_validation_subjects": [7],
        "split_counts": {"development_train": 323, "development_validation": 108},
        "parameter_count": parameter_count(SafeSense(variant="temporal")),
        "seed_results": records,
        "environment": runtime_environment(),
        "orchestration": "three seed-isolated fixed-epoch runs consolidated without per-seed validation checkpoint selection",
        "even_subjects_used": False,
    }
    (output / "development.json").write_text(json.dumps(development, indent=2) + "\n", encoding="utf-8")
    (output / "result.json").write_text(json.dumps(development, indent=2) + "\n", encoding="utf-8")
    robustness = run_safesense_development_robustness(
        prepared_dir,
        output_dir,
        f"{output_dir}/robustness",
        device="cuda",
    )
    result = {"development": development, "robustness": robustness}
    (output / "comparison-ready.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60, volumes={"/results": results_volume})
def safesense_robust_shared27_adapter_smoke() -> dict[str, object]:
    """Smoke the fixed shared adapter replay around the selected epoch-33 temporal base."""
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_robust_adapter_fixed_epoch

    result = run_safesense_robust_adapter_fixed_epoch(
        "/root/fusion-prepared",
        "/results/safesense-robust-fixed33-shared27-adapter-smoke",
        "/results/safesense-temporal-fixed33-selected-development-complete",
        adapter_epoch=1,
        config=SafeSenseTrainingConfig(epochs=1, patience=1),
        seeds=(11,),
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 3, volumes={"/results": results_volume})
def safesense_robust_shared27_adapter_seed(seed: int) -> dict[str, object]:
    """Replay one seed for exactly 27 adapter epochs around its fixed epoch-33 temporal base."""
    if seed not in (11, 22, 33):
        raise ValueError("seed must be one of 11, 22, 33")
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_robust_adapter_fixed_epoch

    result = run_safesense_robust_adapter_fixed_epoch(
        "/root/fusion-prepared",
        f"/results/safesense-robust-fixed33-shared27-adapter-split/seedrun-{seed}",
        "/results/safesense-temporal-fixed33-selected-development-complete",
        adapter_epoch=27,
        config=SafeSenseTrainingConfig(epochs=27, patience=20),
        seeds=(seed,),
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 2, volumes={"/results": results_volume})
def safesense_robust_shared27_adapter_consolidate() -> dict[str, object]:
    """Consolidate the fixed-33/shared-27 two-stage development model and run subject-7 robustness."""
    import json
    import shutil
    from pathlib import Path

    from safesense.robustness import run_safesense_development_robustness
    from safesense.safesense import runtime_environment

    split_root = Path("/results/safesense-robust-fixed33-shared27-adapter-split")
    output = Path("/results/safesense-robust-fixed33-shared27-adapter-complete")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    source_metadata = None
    for seed in (11, 22, 33):
        run_dir = split_root / f"seedrun-{seed}"
        recorded = json.loads((run_dir / "development.json").read_text(encoding="utf-8"))
        if recorded.get("training_mode") != "fixed_epoch_frozen_temporal_backbone_robust_adapter" or recorded.get("seeds") != [seed]:
            raise RuntimeError(f"unexpected shared-27 adapter seed artifact for {seed}")
        if int(recorded.get("frozen_epoch", 0)) != 27:
            raise RuntimeError(f"adapter seed {seed} did not use the fixed shared epoch 27")
        if recorded.get("temporal_run_dir") != "/results/safesense-temporal-fixed33-selected-development-complete":
            raise RuntimeError(f"adapter seed {seed} was not initialized from the selected fixed-33 temporal base")
        record = recorded["seed_results"][0]
        records.append(record)
        source_metadata = recorded
        source = run_dir / f"seed-{seed}"
        destination = output / f"seed-{seed}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    if source_metadata is None:
        raise RuntimeError("no shared-27 adapter metadata found")
    development = {
        **{key: source_metadata[key] for key in (
            "model", "workflow", "status", "variant", "training_mode", "temporal_run_dir",
            "config", "development_train_subjects", "development_validation_subjects", "split_counts",
            "parameter_count", "adapter_trainable_parameter_count", "shared_temporal_backbone_frozen",
            "inference_only_topology_coverage_gate", "adapter_clean_tolerance", "adapter_clean_score_weight",
            "effective_gate_center", "effective_gate_sharpness", "clean_view_weight",
            "robust_temporal_corrupted_weight", "robust_structured_corrupted_weight",
            "reconstruction_weight", "feature_distill_weight", "correction_gate_auxiliary_weight",
            "topology_auxiliary_weight", "lag_auxiliary_ce_weight", "consistency_weight",
            "auxiliary_ce_weight_applied",
        )},
        "seeds": [11, 22, 33],
        "best_epochs": [27, 27, 27],
        "frozen_epoch": 27,
        "epoch_rule": "shared adapter epoch selected by mean robustness-aware subject-7 score with all seed clean floors satisfied",
        "seed_results": records,
        "environment": runtime_environment(),
        "orchestration": "selected temporal epoch 33 replay followed by fixed shared robust adapter epoch 27",
        "even_subjects_used": False,
    }
    (output / "development.json").write_text(json.dumps(development, indent=2) + "\n", encoding="utf-8")
    (output / "result.json").write_text(json.dumps(development, indent=2) + "\n", encoding="utf-8")
    robustness = run_safesense_development_robustness(
        "/root/fusion-prepared",
        output,
        output / "robustness",
        device="cuda",
    )
    result = {"development": development, "robustness": robustness}
    (output / "comparison-ready.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 20, volumes={"/results": results_volume})
def safesense_conditional_ensemble_smoke() -> dict[str, object]:
    """Smoke the subject-7 clean fast path of the conditional robust ensemble."""
    import json
    from pathlib import Path
    import numpy as np
    import torch

    from safesense.robustness import base_temporal_masks
    from safesense.safesense import (
        SAFESENSE_CONDITIONAL_SEEDS,
        _feature_arrays,
        load_safesense_conditional_ensemble,
        parameter_count,
        safesense_splits,
    )

    prepared = Path("/root/fusion-prepared")
    run_dir = Path("/results/safesense-robust-fixed33-shared27-adapter-complete")
    primary = run_dir / f"seed-{SAFESENSE_CONDITIONAL_SEEDS[0]}/checkpoint.pt"
    auxiliary = run_dir / f"seed-{SAFESENSE_CONDITIONAL_SEEDS[1]}/checkpoint.pt"
    model = load_safesense_conditional_ensemble(primary, auxiliary).cuda().eval()
    fused = np.load(prepared / "skeleton_train_features.npy")
    index = int(safesense_splits(prepared).development_valid[0])
    raw = fused[index:index + 1]
    skeleton, imu, _ = _feature_arrays(raw)
    masks = base_temporal_masks(raw)
    pair_mask = np.stack((masks["skeleton"], masks["imu"]), axis=-1).astype(np.float32)
    with np.load(run_dir / "seed-11/stats.npz") as stats:
        skeleton = ((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32)
        imu = ((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32)
    skeleton *= pair_mask[..., 0, None]; imu *= pair_mask[..., 1, None]
    with torch.inference_mode():
        logits = model(torch.from_numpy(skeleton).cuda(), torch.from_numpy(imu).cuda(), torch.from_numpy(pair_mask).cuda())
    route = model.auxiliary_route
    result = {
        "model": "SafeSense", "workflow": "conditional_ensemble_development_smoke", "status": "completed",
        "primary_seed": 11, "auxiliary_seed": 22,
        "parameter_count": parameter_count(model),
        "checkpoint_bytes_fp32_total": int(primary.stat().st_size + auxiliary.stat().st_size),
        "output_shape": list(logits.shape), "finite": bool(torch.isfinite(logits).all()),
        "auxiliary_triggered": bool(route.any()) if route is not None else None,
        "development_validation_subjects": [7], "even_subject_arrays_loaded": False,
        "gpu": torch.cuda.get_device_name(0),
    }
    output = Path("/results/safesense-conditional-ensemble-smoke"); output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    results_volume.commit(); return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 2, volumes={"/results": results_volume})
def safesense_conditional_ensemble_development() -> dict[str, object]:
    """Run subject-7 conditional robustness and clean/routed latency benchmarks."""
    import json
    from pathlib import Path
    from safesense.efficiency import benchmark_conditional_ensemble_development
    from safesense.robustness import run_safesense_conditional_ensemble_development

    robust_dir = "/results/safesense-robust-fixed33-shared27-adapter-complete"
    output = Path("/results/safesense-conditional-ensemble-development"); output.mkdir(parents=True, exist_ok=True)
    robustness = run_safesense_conditional_ensemble_development(
        "/root/fusion-prepared", robust_dir, output / "robustness", device="cuda"
    )
    fusion_record = json.loads(Path(
        "/results/fusion-paired/seed-11/fusion_gcn_reproduction_seed_11.json"
    ).read_text(encoding="utf-8"))
    fusion_checkpoint = fusion_record.get("final_checkpoint")
    if not fusion_checkpoint:
        raise RuntimeError("Fusion-GCN result does not contain a final checkpoint")
    efficiency = benchmark_conditional_ensemble_development(
        robust_dir, "/root/fusion-prepared", "/root/fusion-gcn", fusion_checkpoint,
        output / "efficiency", device="cuda"
    )
    result = {"robustness": robustness, "efficiency": efficiency}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    results_volume.commit(); return result


@app.function(image=image, gpu="T4", timeout=60 * 20, volumes={"/results": results_volume})
def safesense_early_exit_smoke() -> dict[str, object]:
    """Smoke clean fast-path and routed-noise behavior of the sparse early-exit candidate."""
    import json
    from pathlib import Path
    import numpy as np
    import torch

    from safesense.robustness import base_temporal_masks
    from safesense.safesense import (
        SAFESENSE_EARLY_EXIT_SEEDS,
        _feature_arrays,
        load_safesense_early_exit_ensemble,
        parameter_count,
        safesense_splits,
    )

    prepared = Path("/root/fusion-prepared")
    run_dir = Path("/results/safesense-robust-fixed33-shared27-adapter-complete")
    primary = run_dir / f"seed-{SAFESENSE_EARLY_EXIT_SEEDS[0]}/checkpoint.pt"
    auxiliary_source = run_dir / f"seed-{SAFESENSE_EARLY_EXIT_SEEDS[1]}/checkpoint.pt"
    model = load_safesense_early_exit_ensemble(primary, auxiliary_source).cuda().eval()
    fused = np.load(prepared / "skeleton_train_features.npy")
    index = int(safesense_splits(prepared).development_valid[0])
    raw = fused[index:index + 1]
    skeleton, imu, _ = _feature_arrays(raw)
    masks = base_temporal_masks(raw)
    pair_mask = np.stack((masks["skeleton"], masks["imu"]), axis=-1).astype(np.float32)
    with np.load(run_dir / "seed-11/stats.npz") as stats:
        skeleton = ((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32)
        imu = ((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32)
    skeleton *= pair_mask[..., 0, None]
    imu *= pair_mask[..., 1, None]
    skeleton_tensor = torch.from_numpy(skeleton).cuda()
    imu_tensor = torch.from_numpy(imu).cuda()
    mask_tensor = torch.from_numpy(pair_mask).cuda()
    with torch.inference_mode():
        clean_logits = model(skeleton_tensor, imu_tensor, mask_tensor)
    clean_route = model.auxiliary_route
    noisy_imu = imu_tensor.clone()
    noisy_imu[:, ::16, :3] += torch.arange(8, device=noisy_imu.device, dtype=noisy_imu.dtype)[None, :, None] * 20.0
    with torch.inference_mode():
        noisy_logits = model(skeleton_tensor, noisy_imu, mask_tensor)
    noise_route = model.auxiliary_route
    if clean_route is None or bool(clean_route.any()):
        raise RuntimeError("early-exit smoke routed the clean sample")
    if noise_route is None or not bool(noise_route.all()):
        raise RuntimeError("early-exit smoke failed to route synthetic noise")
    output = Path("/results/safesense-early-exit-smoke")
    output.mkdir(parents=True, exist_ok=True)
    auxiliary_checkpoint = output / "seed-22-branch-expert.pt"
    torch.save({name: value.detach().cpu() for name, value in model.auxiliary.state_dict().items()}, auxiliary_checkpoint)
    result = {
        "model": "SafeSense",
        "workflow": "early_exit_development_smoke",
        "status": "completed",
        "primary_seed": 11,
        "auxiliary_seed": 22,
        "parameter_count": parameter_count(model),
        "auxiliary_parameter_count": parameter_count(model.auxiliary),
        "checkpoint_bytes_fp32_total": int(primary.stat().st_size + auxiliary_checkpoint.stat().st_size),
        "clean_output_shape": list(clean_logits.shape),
        "noise_output_shape": list(noisy_logits.shape),
        "finite": bool(torch.isfinite(clean_logits).all() and torch.isfinite(noisy_logits).all()),
        "clean_auxiliary_triggered": False,
        "noise_auxiliary_triggered": True,
        "development_validation_subjects": [7],
        "even_subject_arrays_loaded": False,
        "gpu": torch.cuda.get_device_name(0),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 2, volumes={"/results": results_volume})
def safesense_early_exit_development() -> dict[str, object]:
    """Run subject-7 early-exit robustness plus strict clean/routed T4 efficiency checks."""
    import json
    from pathlib import Path

    from safesense.efficiency import benchmark_early_exit_development
    from safesense.robustness import run_safesense_early_exit_development

    robust_dir = "/results/safesense-robust-fixed33-shared27-adapter-complete"
    output = Path("/results/safesense-early-exit-development")
    output.mkdir(parents=True, exist_ok=True)
    retained = json.loads(Path(
        "/results/safesense-conditional-ensemble-development/robustness/result.json"
    ).read_text(encoding="utf-8"))
    reference = retained.get("conditional_aggregate")
    if not isinstance(reference, dict):
        raise RuntimeError("retained conditional development aggregate is unavailable")
    robustness = run_safesense_early_exit_development(
        "/root/fusion-prepared", robust_dir, output / "robustness",
        device="cuda", reference_aggregate=reference,
    )
    fusion_record = json.loads(Path(
        "/results/fusion-paired/seed-11/fusion_gcn_reproduction_seed_11.json"
    ).read_text(encoding="utf-8"))
    fusion_checkpoint = fusion_record.get("final_checkpoint")
    if not fusion_checkpoint:
        raise RuntimeError("Fusion-GCN result does not contain a final checkpoint")
    efficiency = benchmark_early_exit_development(
        robust_dir, "/root/fusion-prepared", "/root/fusion-gcn", fusion_checkpoint,
        output / "efficiency", device="cuda",
    )
    reference_checks = robustness.get("no_regression_vs_retained_conditional") or {}
    selected = (
        bool(reference_checks.get("all_passed"))
        and bool(efficiency.get("all_latency_checks_passed"))
        and float(robustness["early_exit_aggregate"]["clean_accuracy_mean"]) >= 0.99
    )
    result = {
        "status": "completed",
        "workflow": "early_exit_branch_expert_development",
        "decision": "selected_development_candidate" if selected else "rejected",
        "robustness": robustness,
        "efficiency": efficiency,
        "selection_checks": {
            "clean_accuracy_at_least_99_percent": float(robustness["early_exit_aggregate"]["clean_accuracy_mean"]) >= 0.99,
            "no_regression_vs_retained_conditional": bool(reference_checks.get("all_passed")),
            "clean_and_routed_median_p95_latency_beat_fusion": bool(efficiency.get("all_latency_checks_passed")),
            "all_passed": selected,
        },
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 4, volumes={"/results": results_volume})
def safesense_early_exit_final_comparison() -> dict[str, object]:
    """Run the one frozen post-hoc even-subject comparison of the development-selected early exit."""
    import json
    from pathlib import Path
    import numpy as np

    from safesense.efficiency import benchmark_early_exit_posthoc
    from safesense.robustness import run_safesense_early_exit_posthoc

    final_dir = Path("/results/safesense-robust-two-stage-final-complete")
    output = Path("/results/safesense-early-exit-final-comparison")
    output.mkdir(parents=True, exist_ok=True)
    robustness = run_safesense_early_exit_posthoc(
        "/root/fusion-prepared",
        final_dir,
        "/root/fusion-gcn",
        "/results/fusion-paired",
        output / "robustness",
        device="cuda",
    )
    results_volume.commit()
    fusion_record = json.loads(Path(
        "/results/fusion-paired/seed-11/fusion_gcn_reproduction_seed_11.json"
    ).read_text(encoding="utf-8"))
    fusion_checkpoint = fusion_record.get("final_checkpoint")
    if not fusion_checkpoint:
        raise RuntimeError("Fusion-GCN seed-11 result has no final checkpoint")
    efficiency = benchmark_early_exit_posthoc(
        final_dir,
        "/root/fusion-prepared",
        "/root/fusion-gcn",
        fusion_checkpoint,
        output / "efficiency",
        device="cuda",
    )
    results_volume.commit()

    previous = json.loads(Path(
        "/results/safesense-robust-two-stage-final-comparison/result.json"
    ).read_text(encoding="utf-8"))
    previous_safe = previous["robustness"]["model_aggregates"]["SafeSense"]
    safe = robustness["model_aggregates"]["SafeSense early exit"]
    fusion = robustness["model_aggregates"]["Fusion-GCN"]
    scenarios: dict[str, dict[str, list[float]]] = {}
    for record in robustness["metrics"]:
        scenarios.setdefault(str(record["scenario"]), {}).setdefault(str(record["model"]), []).append(float(record["accuracy"]))
    scenario_delta_vs_fusion = {
        name: float(np.mean(models["SafeSense early exit"])) - float(np.mean(models["Fusion-GCN"]))
        for name, models in scenarios.items()
    }
    summary = {
        "model": "SafeSense",
        "workflow": "early_exit_branch_expert_frozen_posthoc_even_subject_comparison",
        "status": "completed",
        "selection_frozen_before_even_subject_evaluation": True,
        "no_posthoc_retuning_permitted": True,
        "development_selection_artifact": "research/early-exit-development-results.json",
        "training_subjects": [1, 3, 5, 7],
        "posthoc_test_subjects": [2, 4, 6, 8],
        "robustness": robustness,
        "efficiency": efficiency,
        "comparison_vs_fusion_gcn": {
            "clean_accuracy_delta": float(safe["clean_accuracy_mean"]) - float(fusion["clean_accuracy_mean"]),
            "clean_macro_f1_delta": float(safe["clean_macro_f1_mean"]) - float(fusion["clean_macro_f1_mean"]),
            "category_accuracy_delta": {
                category: float(safe["category_accuracy_mean"][category]) - float(fusion["category_accuracy_mean"][category])
                for category in safe["category_accuracy_mean"]
            },
            "category_macro_f1_delta": {
                category: float(safe["category_macro_f1_mean"][category]) - float(fusion["category_macro_f1_mean"][category])
                for category in safe["category_macro_f1_mean"]
            },
            "equal_category_corrupted_accuracy_delta": float(safe["equal_category_corrupted_accuracy_mean"]) - float(fusion["equal_category_corrupted_accuracy_mean"]),
            "equal_category_corrupted_macro_f1_delta": float(safe["equal_category_corrupted_macro_f1_mean"]) - float(fusion["equal_category_corrupted_macro_f1_mean"]),
            "scenario_accuracy_delta": scenario_delta_vs_fusion,
        },
        "comparison_vs_previous_33_27_safesense": {
            "clean_accuracy_delta": float(safe["clean_accuracy_mean"]) - float(previous_safe["clean_accuracy_mean"]),
            "clean_macro_f1_delta": float(safe["clean_macro_f1_mean"]) - float(previous_safe["clean_macro_f1_mean"]),
            "category_accuracy_delta": {
                category: float(safe["category_accuracy_mean"][category]) - float(previous_safe["category_accuracy_mean"][category])
                for category in safe["category_accuracy_mean"]
            },
            "equal_category_corrupted_accuracy_delta": float(safe["equal_category_corrupted_accuracy_mean"]) - float(previous_safe["equal_category_corrupted_accuracy_mean"]),
        },
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    return summary



@app.function(image=image, gpu="T4", timeout=60 * 60 * 6, volumes={"/results": results_volume})
def safesense_odd_fold_crossval(validation_subject: int) -> dict[str, object]:
    """Replay temporal-33/adapter-27 on one leave-one-odd-subject-out development fold."""
    if validation_subject not in (1, 3, 5, 7):
        raise ValueError("validation_subject must be one of 1, 3, 5, 7")
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_odd_fold_two_stage_crossval

    result = run_safesense_odd_fold_two_stage_crossval(
        "/root/fusion-prepared",
        f"/results/safesense-odd-fold-crossval/fold-{validation_subject}",
        temporal_epoch=33,
        adapter_epoch=27,
        config=SafeSenseTrainingConfig(epochs=33, patience=20),
        seeds=(11, 22, 33),
        validation_subjects=(validation_subject,),
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 6, volumes={"/results": results_volume})
def safesense_odd_fold_style_crossval(validation_subject: int) -> dict[str, object]:
    """Test training-only subject/sensor style randomization on one odd validation fold."""
    if validation_subject not in (1, 3, 5, 7):
        raise ValueError("validation_subject must be one of 1, 3, 5, 7")
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_odd_fold_two_stage_crossval

    result = run_safesense_odd_fold_two_stage_crossval(
        "/root/fusion-prepared",
        f"/results/safesense-odd-fold-style-crossval/fold-{validation_subject}",
        temporal_epoch=33,
        adapter_epoch=27,
        config=SafeSenseTrainingConfig(epochs=33, patience=20),
        seeds=(11, 22, 33),
        validation_subjects=(validation_subject,),
        subject_style_training=True,
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 2, volumes={"/results": results_volume})
def safesense_factorized_swa_odd_fold(validation_subject: int, seed: int = 22) -> dict[str, object]:
    """Train/evaluate original factorized model with fixed late SWA and live progress."""
    if validation_subject not in (1,3,5,7) or seed not in (11,22,33): raise ValueError("unsupported validation subject or seed")
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_factorized_swa_odd_fold
    result=run_safesense_factorized_swa_odd_fold("/root/fusion-prepared",f"/results/safesense-factorized-swa-odd-fold/fold-{validation_subject}/seed-{seed}",validation_subject=validation_subject,epochs=60,swa_start_epoch=45,config=SafeSenseTrainingConfig(epochs=60,patience=60,batch_size=32),seeds=(seed,),progress_interval_seconds=30.0,device="cuda")
    results_volume.commit(); return result


@app.function(image=image, gpu="T4", timeout=60 * 30, volumes={"/results": results_volume})
def safesense_factorized_swa_smoke() -> dict[str, object]:
    """Two-epoch T4 smoke for fixed late-SWA collection and telemetry."""
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_factorized_swa_odd_fold
    result=run_safesense_factorized_swa_odd_fold("/root/fusion-prepared","/results/safesense-factorized-swa-smoke",validation_subject=7,epochs=2,swa_start_epoch=2,config=SafeSenseTrainingConfig(epochs=2,patience=2,batch_size=32),seeds=(11,),progress_interval_seconds=5.0,device="cuda")
    results_volume.commit(); return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 2, volumes={"/results": results_volume})
def safesense_factorized_swa_final() -> dict[str, object]:
    """Train frozen seed-22 late-SWA on all odd subjects, then evaluate official even subjects once."""
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_factorized_swa_final
    result=run_safesense_factorized_swa_final("/root/fusion-prepared","/results/safesense-factorized-swa-final",epochs=60,swa_start_epoch=45,seed=22,config=SafeSenseTrainingConfig(epochs=60,patience=60,batch_size=32),progress_interval_seconds=30.0,device="cuda")
    results_volume.commit(); return result



@app.function(image=image, gpu="T4", timeout=60 * 60 * 2, volumes={"/results": results_volume})
def safesense_factorized_odd_fold(validation_subject: int, seed: int = 22) -> dict[str, object]:
    """Fully train and evaluate one factorized odd fold/seed with live progress."""
    if validation_subject not in (1, 3, 5, 7) or seed not in (11, 22, 33):
        raise ValueError("unsupported validation subject or seed")
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_factorized_odd_fold

    result = run_safesense_factorized_odd_fold(
        "/root/fusion-prepared",
        f"/results/safesense-factorized-odd-fold/fold-{validation_subject}/seed-{seed}",
        validation_subject=validation_subject,
        epochs=60,
        config=SafeSenseTrainingConfig(epochs=60, patience=60),
        seeds=(seed,),
        progress_interval_seconds=30.0,
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 30, volumes={"/results": results_volume})
def safesense_factorized_smoke() -> dict[str, object]:
    """One-epoch T4 smoke for the factorized model and progress telemetry."""
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_factorized_odd_fold

    result = run_safesense_factorized_odd_fold(
        "/root/fusion-prepared",
        "/results/safesense-factorized-smoke",
        validation_subject=7,
        epochs=1,
        config=SafeSenseTrainingConfig(epochs=1, patience=1),
        seeds=(11,),
        progress_interval_seconds=5.0,
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 4, volumes={"/results": results_volume})
def safesense_body_canonical_fixed(validation_subject: int = 3, seed: int = 22, temporal_epoch: int = 55, adapter_epoch: int = 27) -> dict[str, object]:
    """Evaluate one fixed body-canonical two-stage odd-fold seed without even-subject access."""
    if validation_subject not in (1, 3, 5, 7) or seed not in (11, 22, 33):
        raise ValueError("unsupported validation subject or seed")
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_odd_fold_two_stage_crossval

    result = run_safesense_odd_fold_two_stage_crossval(
        "/root/fusion-prepared",
        f"/results/safesense-body-canonical-fixed/t{temporal_epoch}-a{adapter_epoch}/fold-{validation_subject}/seed-{seed}",
        temporal_epoch=temporal_epoch,
        adapter_epoch=adapter_epoch,
        config=SafeSenseTrainingConfig(epochs=temporal_epoch, patience=temporal_epoch),
        seeds=(seed,),
        validation_subjects=(validation_subject,),
        subject_style_training=True,
        body_canonicalization=True,
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 4, volumes={"/results": results_volume})
def safesense_body_canonical_temporal_history(validation_subject: int = 3, seed: int = 22) -> dict[str, object]:
    """Record 60-epoch body-canonical temporal history for one odd fold and seed."""
    if validation_subject not in (1, 3, 5, 7) or seed not in (11, 22, 33):
        raise ValueError("unsupported validation subject or seed")
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_odd_fold_temporal_history

    result = run_safesense_odd_fold_temporal_history(
        "/root/fusion-prepared",
        f"/results/safesense-body-canonical-temporal-history/fold-{validation_subject}/seed-{seed}",
        validation_subject=validation_subject,
        max_epoch=60,
        config=SafeSenseTrainingConfig(epochs=60, patience=60),
        seeds=(seed,),
        subject_style_training=True,
        body_canonicalization=True,
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 5, volumes={"/results": results_volume})
def safesense_body_canonical_adapter_history(validation_subject: int = 3, seed: int = 22) -> dict[str, object]:
    """Record a clean-first 40-epoch robust-adapter history around canonical temporal55."""
    if validation_subject not in (1, 3, 5, 7) or seed not in (11, 22, 33):
        raise ValueError("unsupported validation subject or seed")
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_odd_fold_adapter_history

    result = run_safesense_odd_fold_adapter_history(
        "/root/fusion-prepared",
        f"/results/safesense-body-canonical-adapter-history/fold-{validation_subject}/seed-{seed}",
        validation_subject=validation_subject,
        temporal_epoch=55,
        max_adapter_epoch=40,
        config=SafeSenseTrainingConfig(epochs=55, patience=55),
        seeds=(seed,),
        subject_style_training=True,
        body_canonicalization=True,
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 3, volumes={"/results": results_volume})
def safesense_body_canonical_screen(validation_subject: int = 3) -> dict[str, object]:
    """Fast seed-22 odd-fold screen for body-centric skeleton canonicalization plus subject style."""
    if validation_subject not in (1, 3, 5, 7):
        raise ValueError("validation_subject must be one of 1, 3, 5, 7")
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_odd_fold_two_stage_crossval

    result = run_safesense_odd_fold_two_stage_crossval(
        "/root/fusion-prepared",
        f"/results/safesense-body-canonical-screen/fold-{validation_subject}",
        temporal_epoch=33,
        adapter_epoch=27,
        config=SafeSenseTrainingConfig(epochs=33, patience=20),
        seeds=(22,),
        validation_subjects=(validation_subject,),
        subject_style_training=True,
        body_canonicalization=True,
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 4, volumes={"/results": results_volume})
def safesense_body_canonical_confirm(validation_subject: int) -> dict[str, object]:
    """Confirm body-canonical odd-fold stability on seeds 11 and 33 after seed-22 screening."""
    if validation_subject not in (1, 3, 5, 7):
        raise ValueError("validation_subject must be one of 1, 3, 5, 7")
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_odd_fold_two_stage_crossval

    result = run_safesense_odd_fold_two_stage_crossval(
        "/root/fusion-prepared",
        f"/results/safesense-body-canonical-confirm/fold-{validation_subject}",
        temporal_epoch=33,
        adapter_epoch=27,
        config=SafeSenseTrainingConfig(epochs=33, patience=20),
        seeds=(11, 33),
        validation_subjects=(validation_subject,),
        subject_style_training=True,
        body_canonicalization=True,
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 6, volumes={"/results": results_volume})
def safesense_style_final_seed22() -> dict[str, object]:
    """Replay the odd-fold-selected subject-style SafeSense with seed 22 on all odd subjects."""
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_robust_two_stage_final

    result = run_safesense_robust_two_stage_final(
        "/root/fusion-prepared",
        "/results/safesense-style-final-seed22",
        temporal_epoch=33,
        adapter_epoch=27,
        config=SafeSenseTrainingConfig(epochs=33, patience=20),
        seeds=(22,),
        subject_style_training=True,
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60, volumes={"/results": results_volume})
def safesense_robust_two_stage_final_smoke() -> dict[str, object]:
    """Smoke the frozen two-stage final runner with one epoch per stage."""
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_robust_two_stage_final

    result = run_safesense_robust_two_stage_final(
        "/root/fusion-prepared",
        "/results/safesense-robust-two-stage-final-smoke",
        temporal_epoch=1,
        adapter_epoch=1,
        config=SafeSenseTrainingConfig(epochs=1, patience=1),
        seeds=(11,),
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 5, volumes={"/results": results_volume})
def safesense_robust_two_stage_final_seed(seed: int) -> dict[str, object]:
    """Train/evaluate one frozen final seed with temporal-33 plus adapter-27."""
    if seed not in (11, 22, 33):
        raise ValueError("seed must be one of 11, 22, 33")
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_robust_two_stage_final

    result = run_safesense_robust_two_stage_final(
        "/root/fusion-prepared",
        f"/results/safesense-robust-two-stage-final-split/seedrun-{seed}",
        temporal_epoch=33,
        adapter_epoch=27,
        config=SafeSenseTrainingConfig(epochs=33, patience=20),
        seeds=(seed,),
        device="cuda",
    )
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 2, volumes={"/results": results_volume})
def safesense_robust_two_stage_final_consolidate() -> dict[str, object]:
    """Consolidate the three frozen two-stage final seeds without retraining or selection."""
    import json
    import shutil
    from pathlib import Path
    import numpy as np

    from safesense.safesense import runtime_environment

    split_root = Path("/results/safesense-robust-two-stage-final-split")
    output = Path("/results/safesense-robust-two-stage-final-complete")
    output.mkdir(parents=True, exist_ok=True)
    seed_results = []
    training_records = []
    source_metadata = None
    for seed in (11, 22, 33):
        run_dir = split_root / f"seedrun-{seed}"
        recorded = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if recorded.get("training_mode") != "two_stage_fixed_temporal_and_adapter" or recorded.get("seeds") != [seed]:
            raise RuntimeError(f"unexpected frozen two-stage final seed artifact for {seed}")
        if int(recorded.get("temporal_epoch", 0)) != 33 or int(recorded.get("adapter_epoch", 0)) != 27:
            raise RuntimeError(f"final seed {seed} did not use frozen temporal-33/adapter-27 policy")
        seed_results.extend(recorded["seed_results"])
        training_records.extend(recorded["training_records"])
        source_metadata = recorded
        source = run_dir / f"seed-{seed}"
        destination = output / f"seed-{seed}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        seed_results[-1]["checkpoint"] = str(destination / "checkpoint.pt")
        training_records[-1]["checkpoint"] = str(destination / "checkpoint.pt")
        training_records[-1]["temporal_checkpoint"] = str(destination / "temporal-checkpoint.pt")
    if source_metadata is None:
        raise RuntimeError("no frozen two-stage final metadata found")
    result = {
        **{key: source_metadata[key] for key in (
            "model", "workflow", "status", "variant", "training_mode", "temporal_epoch", "adapter_epoch",
            "epoch_rule", "config", "training_subjects", "official_test_subjects", "split_counts",
            "parameter_count", "adapter_trainable_parameter_count",
            "shared_temporal_backbone_frozen_during_adapter_training", "inference_only_topology_coverage_gate",
            "clean_view_weight", "robust_temporal_corrupted_weight", "robust_structured_corrupted_weight",
            "reconstruction_weight", "feature_distill_weight", "correction_gate_auxiliary_weight",
            "topology_auxiliary_weight", "lag_auxiliary_ce_weight", "consistency_weight",
            "auxiliary_ce_weight_applied", "held_out_evaluation_after_all_training_completed",
        )},
        "seeds": [11, 22, 33],
        "training_records": training_records,
        "seed_results": seed_results,
        "aggregate": {
            "accuracy_mean": float(np.mean([item["accuracy"] for item in seed_results])),
            "accuracy_std_population": float(np.std([item["accuracy"] for item in seed_results], ddof=0)),
            "macro_f1_mean": float(np.mean([item["macro_f1"] for item in seed_results])),
            "macro_f1_std_population": float(np.std([item["macro_f1"] for item in seed_results], ddof=0)),
        },
        "stats": str(output / "seed-11/stats.npz"),
        "environment": runtime_environment(),
        "orchestration": "three seed-isolated frozen temporal-33/adapter-27 final runs consolidated without selection",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 4, volumes={"/results": results_volume})
def safesense_robust_two_stage_final_comparison() -> dict[str, object]:
    """Compare the frozen two-stage SafeSense final model with reproduced Fusion-GCN."""
    import json
    from pathlib import Path
    import numpy as np

    from safesense.efficiency import benchmark_frozen_models
    from safesense.robustness import run_robustness_evaluation

    final_dir = Path("/results/safesense-robust-two-stage-final-complete")
    comparison_dir = Path("/results/safesense-robust-two-stage-final-comparison")
    final = json.loads((final_dir / "result.json").read_text(encoding="utf-8"))
    robustness = run_robustness_evaluation(
        "/root/fusion-prepared",
        final_dir,
        "/root/fusion-gcn",
        "/results/fusion-paired",
        comparison_dir / "robustness",
        device="cuda",
        safesense_variant="robust",
    )
    results_volume.commit()
    fusion_recorded = json.loads(Path("/results/fusion-paired/seed-11/fusion_gcn_reproduction_seed_11.json").read_text(encoding="utf-8"))
    fusion_checkpoint = fusion_recorded.get("final_checkpoint")
    if not fusion_checkpoint:
        raise RuntimeError("Fusion-GCN seed-11 result has no final checkpoint")
    efficiency = benchmark_frozen_models(
        final_dir / "seed-11/checkpoint.pt",
        "/root/fusion-prepared",
        final_dir / "seed-11/stats.npz",
        "/root/fusion-gcn",
        fusion_checkpoint,
        comparison_dir / "efficiency",
        device="cuda",
        safesense_variant="robust",
    )
    results_volume.commit()
    safe = robustness["model_aggregates"]["SafeSense"]
    fusion = robustness["model_aggregates"]["Fusion-GCN"]
    scenarios: dict[str, dict[str, list[float]]] = {}
    for record in robustness["metrics"]:
        bucket = scenarios.setdefault(str(record["scenario"]), {}).setdefault(str(record["model"]), [])
        bucket.append(float(record["accuracy"]))
    scenario_delta = {
        name: float(np.mean(models["SafeSense"])) - float(np.mean(models["Fusion-GCN"]))
        for name, models in scenarios.items()
    }
    safe_eff = efficiency["models"]["SafeSense"]
    fusion_eff = efficiency["models"]["Fusion-GCN"]
    summary = {
        "model": "SafeSense",
        "workflow": "robust_two_stage_final_posthoc_comparison",
        "status": "completed",
        "selection_frozen_before_even_subject_evaluation": True,
        "temporal_epoch": 33,
        "adapter_epoch": 27,
        "seeds": [11, 22, 33],
        "training_subjects": [1, 3, 5, 7],
        "posthoc_test_subjects": [2, 4, 6, 8],
        "safesense_final": final,
        "robustness": {
            "model_aggregates": robustness["model_aggregates"],
            "clean_accuracy_delta": float(safe["clean_accuracy_mean"]) - float(fusion["clean_accuracy_mean"]),
            "clean_macro_f1_delta": float(safe["clean_macro_f1_mean"]) - float(fusion["clean_macro_f1_mean"]),
            "category_accuracy_delta": {
                category: float(safe["category_accuracy_mean"][category]) - float(fusion["category_accuracy_mean"][category])
                for category in safe["category_accuracy_mean"]
            },
            "category_macro_f1_delta": {
                category: float(safe["category_macro_f1_mean"][category]) - float(fusion["category_macro_f1_mean"][category])
                for category in safe["category_macro_f1_mean"]
            },
            "equal_category_corrupted_accuracy_delta": float(safe["equal_category_corrupted_accuracy_mean"]) - float(fusion["equal_category_corrupted_accuracy_mean"]),
            "equal_category_corrupted_macro_f1_delta": float(safe["equal_category_corrupted_macro_f1_mean"]) - float(fusion["equal_category_corrupted_macro_f1_mean"]),
            "scenario_accuracy_delta": scenario_delta,
        },
        "efficiency": efficiency,
        "cost_delta": {
            "parameter_reduction_fraction": 1.0 - float(safe_eff["parameters"]) / float(fusion_eff["parameters"]),
            "checkpoint_reduction_fraction": 1.0 - float(safe_eff["checkpoint_bytes_fp32"]) / float(fusion_eff["checkpoint_bytes_fp32"]),
            "median_latency_delta_ms": float(safe_eff["median_ms_model_only"]) - float(fusion_eff["median_ms_model_only"]),
            "fusion_to_safesense_latency_ratio": efficiency["latency_ratio_fusion_to_safesense"],
        },
        "fusion_gcn_context": {
            "same_code_comparison": True,
            "checkpoint_policy": robustness["fusion_checkpoint_policy"],
            "paper_reported_accuracy": 0.9442,
            "paper_result_is_not_same_code": True,
        },
    }
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    return summary


@app.function(image=image, gpu="T4", timeout=60 * 60 * 8, volumes={"/results": results_volume})
def safesense_temporal_final_comparison() -> dict[str, object]:
    """Run the frozen epoch-19 post-hoc SafeSense vs Fusion-GCN comparison."""
    import json
    from pathlib import Path

    import numpy as np

    from safesense.efficiency import benchmark_frozen_models
    from safesense.robustness import run_robustness_evaluation
    from safesense.safesense import SafeSenseTrainingConfig, run_safesense_final

    final_dir = "/results/safesense-temporal-final"
    comparison_dir = Path("/results/safesense-temporal-final-comparison")
    final = run_safesense_final(
        "/root/fusion-prepared",
        final_dir,
        frozen_epoch=19,
        config=SafeSenseTrainingConfig(epochs=19, patience=20),
        seeds=(11, 22, 33),
        variant="temporal",
        device="cuda",
    )
    results_volume.commit()

    robustness = run_robustness_evaluation(
        "/root/fusion-prepared",
        final_dir,
        "/root/fusion-gcn",
        "/results/fusion-paired",
        comparison_dir / "robustness",
        device="cuda",
        safesense_variant="temporal",
    )
    results_volume.commit()

    fusion_result_path = Path("/results/fusion-paired/seed-11/fusion_gcn_reproduction_seed_11.json")
    fusion_recorded = json.loads(fusion_result_path.read_text(encoding="utf-8"))
    fusion_checkpoint = fusion_recorded.get("final_checkpoint")
    if not fusion_checkpoint:
        raise RuntimeError("Fusion-GCN seed-11 result has no final checkpoint")
    efficiency = benchmark_frozen_models(
        Path(final_dir) / "seed-11/checkpoint.pt",
        "/root/fusion-prepared",
        Path(final_dir) / "seed-11/stats.npz",
        "/root/fusion-gcn",
        fusion_checkpoint,
        comparison_dir / "efficiency",
        device="cuda",
        safesense_variant="temporal",
    )
    results_volume.commit()

    model_aggregates = robustness["model_aggregates"]
    safe_aggregate = model_aggregates["SafeSense"]
    fusion_aggregate = model_aggregates["Fusion-GCN"]
    category_accuracy_delta = {
        category: float(safe_aggregate["category_accuracy_mean"][category])
        - float(fusion_aggregate["category_accuracy_mean"][category])
        for category in safe_aggregate["category_accuracy_mean"]
    }
    category_macro_f1_delta = {
        category: float(safe_aggregate["category_macro_f1_mean"][category])
        - float(fusion_aggregate["category_macro_f1_mean"][category])
        for category in safe_aggregate["category_macro_f1_mean"]
    }

    scenario_means: dict[str, dict[str, dict[str, float]]] = {}
    for record in robustness["metrics"]:
        scenario = str(record["scenario"])
        model = str(record["model"])
        bucket = scenario_means.setdefault(scenario, {}).setdefault(model, {"accuracy": [], "macro_f1": []})
        bucket["accuracy"].append(float(record["accuracy"]))
        bucket["macro_f1"].append(float(record["macro_f1"]))
    scenario_comparison = {}
    for scenario, models in scenario_means.items():
        safe_accuracy = float(np.mean(models["SafeSense"]["accuracy"]))
        fusion_accuracy = float(np.mean(models["Fusion-GCN"]["accuracy"]))
        safe_macro_f1 = float(np.mean(models["SafeSense"]["macro_f1"]))
        fusion_macro_f1 = float(np.mean(models["Fusion-GCN"]["macro_f1"]))
        scenario_comparison[scenario] = {
            "safesense_accuracy": safe_accuracy,
            "fusion_gcn_accuracy": fusion_accuracy,
            "accuracy_delta": safe_accuracy - fusion_accuracy,
            "safesense_macro_f1": safe_macro_f1,
            "fusion_gcn_macro_f1": fusion_macro_f1,
            "macro_f1_delta": safe_macro_f1 - fusion_macro_f1,
        }

    safe_efficiency = efficiency["models"]["SafeSense"]
    fusion_efficiency = efficiency["models"]["Fusion-GCN"]
    summary = {
        "model": "SafeSense",
        "workflow": "temporal_final_posthoc_comparison",
        "status": "completed",
        "selection_frozen_before_even_subject_evaluation": True,
        "frozen_epoch": 19,
        "seeds": [11, 22, 33],
        "training_subjects": [1, 3, 5, 7],
        "posthoc_test_subjects": [2, 4, 6, 8],
        "safesense_final": final,
        "robustness": {
            "model_aggregates": model_aggregates,
            "clean_accuracy_delta": float(safe_aggregate["clean_accuracy_mean"]) - float(fusion_aggregate["clean_accuracy_mean"]),
            "clean_macro_f1_delta": float(safe_aggregate["clean_macro_f1_mean"]) - float(fusion_aggregate["clean_macro_f1_mean"]),
            "category_accuracy_delta": category_accuracy_delta,
            "category_macro_f1_delta": category_macro_f1_delta,
            "equal_category_corrupted_accuracy_delta": float(safe_aggregate["equal_category_corrupted_accuracy_mean"])
            - float(fusion_aggregate["equal_category_corrupted_accuracy_mean"]),
            "equal_category_corrupted_macro_f1_delta": float(safe_aggregate["equal_category_corrupted_macro_f1_mean"])
            - float(fusion_aggregate["equal_category_corrupted_macro_f1_mean"]),
            "scenario_comparison": scenario_comparison,
        },
        "efficiency": efficiency,
        "cost_delta": {
            "parameter_reduction_fraction": 1.0 - float(safe_efficiency["parameters"]) / float(fusion_efficiency["parameters"]),
            "checkpoint_reduction_fraction": 1.0 - float(safe_efficiency["checkpoint_bytes_fp32"]) / float(fusion_efficiency["checkpoint_bytes_fp32"]),
            "median_latency_delta_ms": float(safe_efficiency["median_ms_model_only"])
            - float(fusion_efficiency["median_ms_model_only"]),
            "fusion_to_safesense_latency_ratio": efficiency["latency_ratio_fusion_to_safesense"],
        },
        "fusion_gcn_context": {
            "same_code_comparison": True,
            "checkpoint_policy": robustness["fusion_checkpoint_policy"],
            "paper_reported_accuracy": 0.9442,
            "paper_result_is_not_same_code": True,
        },
    }
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    results_volume.commit()
    return summary


@app.function(image=image, gpu="T4", timeout=60 * 60 * 8, volumes={"/results": results_volume})
def robustness_evaluation(
    prepared_dir: str = "/root/fusion-prepared",
    safesense_dir: str = "/results/safesense-final",
    fusion_repo_dir: str = "/root/fusion-gcn",
    fusion_results_dir: str = "/results/fusion-paired",
    output_dir: str = "/results/robustness",
    safesense_variant: str = "full",
) -> dict[str, object]:
    """Evaluate frozen final checkpoints under deterministic corruptions."""
    from safesense.robustness import run_robustness_evaluation

    result = run_robustness_evaluation(
        prepared_dir,
        safesense_dir,
        fusion_repo_dir,
        fusion_results_dir,
        output_dir,
        device="cuda",
        safesense_variant=safesense_variant,
    )
    results_volume.commit()
    return result



@app.function(image=image, gpu="T4", timeout=60 * 30, volumes={"/results": results_volume})
def safesense_factorized_swa_final_efficiency() -> dict[str, object]:
    """Benchmark frozen factorized seed 22 and preserved Fusion-GCN seed 11 on one T4."""
    import json
    from pathlib import Path
    import numpy as np, torch
    from safesense.efficiency import CudaGraphBatch1Inference, benchmark_model, checkpoint_bytes
    from safesense.robustness import base_temporal_masks, load_fusion_final_model
    from safesense.safesense import SafeSenseFactorized, _feature_arrays
    checkpoint=Path("/results/safesense-factorized-swa-final/seed-22/checkpoint.pt"); stats_path=Path("/results/safesense-factorized-swa-final/seed-22/stats.npz")
    safe=SafeSenseFactorized(); safe.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)); safe.eval()
    safe_graph_base=SafeSenseFactorized().cuda(); safe_graph_base.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)); safe_graph_base.eval()
    fusion_checkpoint="/root/fusion-gcn-seed11-final.pt"
    fusion=load_fusion_final_model("/root/fusion-gcn",fusion_checkpoint,torch.device("cuda"))
    raw=np.load("/root/fusion-prepared/skeleton_val_features.npy")[:1]; skeleton,imu,_=_feature_arrays(raw); masks=base_temporal_masks(raw); pair=np.stack((masks["skeleton"],masks["imu"]),axis=-1).astype(np.float32)
    with np.load(stats_path) as stats:
        skeleton=((skeleton-stats["skeleton_mean"])/stats["skeleton_std"]).astype(np.float32); imu=((imu-stats["imu_mean"])/stats["imu_std"]).astype(np.float32)
    skeleton*=pair[...,0,None]; imu*=pair[...,1,None]
    safe_inputs=(torch.from_numpy(skeleton),torch.from_numpy(imu),torch.from_numpy(pair))
    safe_eager=benchmark_model(safe,safe_inputs,device="cuda",warmups=200,iterations=1000,checkpoint_bytes=checkpoint_bytes(checkpoint))
    safe_graph_inputs=tuple(value.cuda() for value in safe_inputs)
    safe_graph=CudaGraphBatch1Inference(safe_graph_base,safe_graph_inputs)
    safe_cudagraph=benchmark_model(safe_graph,safe_graph_inputs,device="cuda",warmups=200,iterations=1000,checkpoint_bytes=checkpoint_bytes(checkpoint))
    fusion_eager=benchmark_model(fusion,(torch.from_numpy(raw),),device="cuda",warmups=200,iterations=1000,checkpoint_bytes=checkpoint_bytes(fusion_checkpoint))
    fusion_graph_inputs=(torch.from_numpy(raw).cuda(),)
    fusion_graph=CudaGraphBatch1Inference(fusion,fusion_graph_inputs)
    fusion_cudagraph=benchmark_model(fusion_graph,fusion_graph_inputs,device="cuda",warmups=200,iterations=1000,checkpoint_bytes=checkpoint_bytes(fusion_checkpoint))
    result={"status":"completed","device":"cuda","batch_size":1,"dtype":"float32","warmups":200,"iterations":1000,"models":{"SafeSense-factorized-SWA-seed22-eager":safe_eager,"SafeSense-factorized-SWA-seed22-cudagraph":safe_cudagraph,"Fusion-GCN-seed11-eager":fusion_eager,"Fusion-GCN-seed11-cudagraph":fusion_cudagraph},"latency_ratio_fusion_to_safesense_eager":fusion_eager["median_ms_model_only"]/safe_eager["median_ms_model_only"],"latency_ratio_fusion_to_safesense_cudagraph":fusion_cudagraph["median_ms_model_only"]/safe_cudagraph["median_ms_model_only"],"latency_ratio_safesense_eager_to_cudagraph":safe_eager["median_ms_model_only"]/safe_cudagraph["median_ms_model_only"],"latency_scope":"model-only forward; CUDA Graph paths include device-to-device input copies into static buffers and graph replay; excludes preprocessing and host-to-device transfer"}
    out=Path("/results/safesense-factorized-swa-final-efficiency"); out.mkdir(parents=True,exist_ok=True); (out/"result.json").write_text(json.dumps(result,indent=2)+"\n"); results_volume.commit(); return result


@app.function(image=image, gpu="T4", timeout=60 * 60 * 2, volumes={"/results": results_volume})
def efficiency_benchmark(
    safesense_checkpoint: str = "/results/safesense-final/seed-11/checkpoint.pt",
    prepared_dir: str = "/root/fusion-prepared",
    safesense_stats_path: str = "/results/safesense-final/seed-11/stats.npz",
    fusion_repo_dir: str = "/root/fusion-gcn",
    fusion_result_json: str = "/results/fusion-paired/seed-11/fusion_gcn_reproduction_seed_11.json",
    output_dir: str = "/results/efficiency",
    safesense_variant: str = "full",
) -> dict[str, object]:
    """Benchmark model-only FP32 T4 latency with 200 warmups and 1000 iterations."""
    import json

    from safesense.efficiency import benchmark_frozen_models

    recorded = json.loads(open(fusion_result_json, encoding="utf-8").read())
    fusion_checkpoint = recorded.get("final_checkpoint")
    if not fusion_checkpoint:
        raise RuntimeError("Fusion-GCN result does not contain a final checkpoint")
    result = benchmark_frozen_models(
        safesense_checkpoint,
        prepared_dir,
        safesense_stats_path,
        fusion_repo_dir,
        fusion_checkpoint,
        output_dir,
        device="cuda",
        safesense_variant=safesense_variant,
    )
    results_volume.commit()
    return result


@app.local_entrypoint()
def main() -> None:
    result = environment_smoke.remote()
    print(result)
