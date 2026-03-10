# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.system_metrics import SystemMetricsMonitor, append_summary_csv
from opentslm.time_series_datasets.TSQADataset import TSQADataset
from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from opentslm.time_series_datasets.simulation.SimulationQADataset import SimulationQADataset
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


REPO_DIR = Path(__file__).resolve().parents[1]


def get_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_optimizer(model, model_type: str, base_lr: float = 2e-4):
    if model_type == "OpenTSLMSP":
        enc_params = [p for p in getattr(model, "encoder").parameters() if p.requires_grad]
        proj_params = [p for p in getattr(model, "projector").parameters() if p.requires_grad]
        param_groups = []
        if enc_params:
            param_groups.append({"params": enc_params, "weight_decay": 0.1})
        if proj_params:
            param_groups.append({"params": proj_params, "weight_decay": 0.1})
        return torch.optim.AdamW(param_groups, lr=base_lr) if param_groups else None

    named_params = list(model.named_parameters())
    trainable = [
        (name, param)
        for name, param in named_params
        if param.requires_grad and not getattr(param, "exclude_from_optimizer", False)
    ]
    params_with_wd = [param for name, param in trainable if "gated_cross_attn" in name]
    params_without_wd = [param for name, param in trainable if "gated_cross_attn" not in name]
    if not params_with_wd and not params_without_wd:
        return None
    return torch.optim.AdamW(
        [
            {"params": params_with_wd, "weight_decay": 0.1},
            {"params": params_without_wd, "weight_decay": 0.0},
        ],
        lr=base_lr,
    )


def instantiate_model(model_name: str, llm_id: str, device: str):
    if model_name == "OpenTSLMFlamingo":
        model = OpenTSLMFlamingo(
            device=device,
            llm_id=llm_id,
            cross_attn_every_n_layers=1,
            gradient_checkpointing=True,
        )
    elif model_name == "OpenTSLMSP":
        model = OpenTSLMSP(llm_id=llm_id, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.to(device)
    return model


def instantiate_dataset(dataset_name: str, eos: str, length: int, num_series: int):
    if dataset_name == "TSQADataset":
        return TSQADataset(split="train", EOS_TOKEN=eos), "TSQA"
    if dataset_name == "HARCoTQADataset":
        return HARCoTQADataset(split="train", EOS_TOKEN=eos), "HAR-CoT"
    if dataset_name == "SleepEDFCoTQADataset":
        return SleepEDFCoTQADataset(split="train", EOS_TOKEN=eos), "SleepEDF-CoT"
    if dataset_name == "ECGQACoTQADataset":
        return (
            ECGQACoTQADataset(
                split="train",
                EOS_TOKEN=eos,
                max_samples=1,
                preload_processed_data=False,
            ),
            "ECG-QA-CoT",
        )
    if dataset_name == "SimulationQADataset":
        return (
            SimulationQADataset(
                split="train",
                EOS_TOKEN=eos,
                length=length,
                num_series=num_series,
            ),
            f"Simulation-L{length}-N{num_series}",
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def train_for_steps(
    model,
    model_name: str,
    dataset,
    steps: int,
    monitor: SystemMetricsMonitor,
) -> float:
    model.train()
    optimizer = build_optimizer(model, model_name)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(batch),
        drop_last=False,
    )

    last_loss = 0.0
    start = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        step_start = time.perf_counter()
        if optimizer:
            optimizer.zero_grad(set_to_none=True)

        loss = model.compute_loss(batch)
        if optimizer and loss.requires_grad:
            loss.backward()
            optimizer.step()

        last_loss = float(loss.detach().item())
        step_duration = time.perf_counter() - step_start
        monitor.mark(
            phase="train_step",
            step=step,
            loss=last_loss,
            step_duration_s=step_duration,
        )
        if step >= steps:
            break

    monitor.mark(
        phase="train_end",
        step=steps,
        train_total_time_s=time.perf_counter() - start,
        final_loss=last_loss,
    )
    return last_loss


def run_inference(
    model,
    dataset,
    batches: int,
    max_new_tokens: int,
    monitor: SystemMetricsMonitor,
) -> Tuple[List[str], List[float]]:
    model.eval()
    outputs: List[str] = []
    latencies: List[float] = []

    with torch.inference_mode():
        for idx in range(min(batches, len(dataset))):
            batch = extend_time_series_to_match_patch_size_and_aggregate([dataset[idx]])
            started = time.perf_counter()
            generated = model.generate(batch, max_new_tokens=max_new_tokens)
            latency = time.perf_counter() - started
            outputs.extend(generated)
            latencies.append(latency)
            monitor.mark(
                phase="inference_step",
                step=idx + 1,
                inference_latency_s=latency,
                generated_chars=len(generated[0]) if generated else 0,
            )

    monitor.mark(
        phase="inference_end",
        step=len(latencies),
        inference_total_time_s=sum(latencies),
        inference_mean_latency_s=(sum(latencies) / len(latencies)) if latencies else 0.0,
    )
    return outputs, latencies


def main():
    parser = argparse.ArgumentParser(
        description="Collect detailed CPU, RAM, GPU, disk, and timing metrics during training and inference."
    )
    parser.add_argument("-llm_id", required=True, help="Hugging Face model id")
    parser.add_argument("--model", required=True, choices=["OpenTSLMFlamingo", "OpenTSLMSP"])
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "TSQADataset",
            "HARCoTQADataset",
            "SleepEDFCoTQADataset",
            "ECGQACoTQADataset",
            "SimulationQADataset",
        ],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--train_steps", type=int, default=10)
    parser.add_argument("--inference_batches", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--sample_interval", type=float, default=0.5)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--num_series", type=int, default=1)
    parser.add_argument("--output_dir", default=str(REPO_DIR / "results" / "system_metrics"))
    args = parser.parse_args()

    device = get_device(args.device)
    model = instantiate_model(args.model, args.llm_id, device)
    dataset, dataset_label = instantiate_dataset(
        args.dataset, model.get_eos_token(), args.length, args.num_series
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{timestamp}_{args.model}_{args.dataset}"

    metadata = {
        "llm_id": args.llm_id,
        "model": args.model,
        "dataset": dataset_label,
        "device": device,
    }

    monitor = SystemMetricsMonitor(
        label=base_name,
        interval_s=args.sample_interval,
        disk_path=output_dir,
        metadata=metadata,
    ).start()
    monitor.mark(phase="setup", dataset_size=len(dataset))

    effective_train_steps = max(1, min(args.train_steps, len(dataset)))
    effective_inference_batches = max(1, min(args.inference_batches, len(dataset)))

    train_loss = train_for_steps(
        model=model,
        model_name=args.model,
        dataset=dataset,
        steps=effective_train_steps,
        monitor=monitor,
    )
    _, latencies = run_inference(
        model=model,
        dataset=dataset,
        batches=effective_inference_batches,
        max_new_tokens=args.max_new_tokens,
        monitor=monitor,
    )
    monitor.stop(
        final_phase="finished",
        final_loss=train_loss,
        mean_inference_latency_s=(sum(latencies) / len(latencies)) if latencies else 0.0,
    )

    samples_path = monitor.to_csv(output_dir / f"{base_name}_samples.csv")
    monitor.to_jsonl(output_dir / f"{base_name}_samples.jsonl")
    summary = monitor.summary().to_dict()
    summary.update(metadata)
    summary["train_steps"] = effective_train_steps
    summary["inference_batches"] = effective_inference_batches
    append_summary_csv(output_dir / "summary.csv", [summary])

    print(f"Detailed samples saved to: {samples_path}")
    print(f"Summary appended to: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
