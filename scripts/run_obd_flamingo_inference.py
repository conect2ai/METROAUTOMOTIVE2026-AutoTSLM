#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = next(
    (path for path in (Path.cwd().resolve(), *Path.cwd().resolve().parents) if (path / "pyproject.toml").exists()),
    Path.cwd().resolve(),
)
for extra_path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "open_flamingo",
    PROJECT_ROOT / "src" / "open_flamingo",
):
    if extra_path.exists() and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from opentslm.system_metrics import SystemMetricsCollector, SystemMetricsMonitor

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


SEED = 42
FEATURE_LABELS = ["Speed", "RPM", "Engine Load", "Throttle Position"]
LABELS = ["economical", "normal", "aggressive", "congested"]
NO_LEAK_PROMPT = """You are shown a time-series plot of vehicle telemetry over a 120-sample window.
This data corresponds to one of two possible activities:
[{label}]
[{dis}]
Your task is to classify the activity based on analysis of the data.
Instructions:
- Begin by analyzing the time-series without assuming a specific label.
- Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.
- Write your rationale as a single, natural paragraph, do not use bullet points, numbered steps, or section headings.
- Do not refer back to the plot or to the act of visual analysis in your rationale; the plot is only for reference but you should reason about the time-series data.
- Do not assume any answer at the beginning, analyze as if you do not yet know which class is correct.
- Do not mention either class label until the final sentence.
- Make sure that your last word is the answer. You MUST end your response with "Answer:"."""


@dataclass
class ProcessSnapshot:
    cpu_time_s: float
    rss_bytes: int | None
    vms_bytes: int | None
    cpu_temp_c: float | None
    cpu_freq_mhz: float | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def detect_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, dtype_name: str | None) -> torch.dtype:
    if dtype_name is None:
        return torch.float32 if device == "cpu" else torch.bfloat16
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]


def build_prompt(row: dict[str, Any]) -> str:
    return NO_LEAK_PROMPT.format(label=row.get("label") or "", dis=row.get("dissimilar_label") or "")


def load_rows(dataset_path: Path, drop_congested: bool) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if drop_congested:
        rows = [row for row in rows if row.get("label") != "congested"]
    for row in rows:
        row["pre_prompt"] = build_prompt(row)
    return rows


def split_rows(rows: list[dict[str, Any]], seed: int, split: str) -> list[dict[str, Any]]:
    if split == "all":
        return rows

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)

    n_rows = len(indices)
    train_end = max(1, int(0.6 * n_rows))
    val_end = max(train_end + 1, int(0.8 * n_rows)) if n_rows > 2 else train_end

    train_rows = [rows[i] for i in indices[:train_end]]
    val_rows = [rows[i] for i in indices[train_end:val_end]]
    test_rows = [rows[i] for i in indices[val_end:]]

    if len(val_rows) == 0 and len(train_rows) > 1:
        val_rows = [train_rows.pop()]
    if len(test_rows) == 0 and len(train_rows) > 1:
        test_rows = [train_rows.pop()]

    if split == "train":
        return train_rows
    if split == "val":
        return val_rows
    if split == "test":
        return test_rows
    raise ValueError(f"Unsupported split: {split}")


def summarize_series(label: str, values: torch.Tensor, n_points: int = 8) -> str:
    values = values.detach().cpu().float()
    n_values = values.numel()
    index = torch.linspace(0, max(n_values - 1, 0), steps=min(n_points, n_values)).long()
    samples = ", ".join(f"{values[i].item():.1f}" for i in index)
    delta = values[-1].item() - values[0].item()
    if delta > 1e-3:
        trend = "overall increasing"
    elif delta < -1e-3:
        trend = "overall decreasing"
    else:
        trend = "roughly flat"
    return (
        f"{label} series with mean {values.mean().item():.2f}, "
        f"std {values.std(unbiased=False).item():.2f}, "
        f"min {values.min().item():.2f}, max {values.max().item():.2f}, "
        f"trend {trend}, sampled points [{samples}]:"
    )


def prepare_sample(row: dict[str, Any]) -> dict[str, Any]:
    time_series = torch.tensor(row["time_series"], dtype=torch.float32)
    return {
        "id": row["id"],
        "pre_prompt": row["pre_prompt"],
        "post_prompt": row["post_prompt"],
        "answer": row["answer"],
        "time_series": time_series,
        "time_series_text": [summarize_series(label, series) for label, series in zip(FEATURE_LABELS, time_series)],
    }


def read_cpu_temp_c() -> float | None:
    thermal_paths = [
        Path("/sys/class/thermal/thermal_zone0/temp"),
        Path("/sys/devices/virtual/thermal/thermal_zone0/temp"),
    ]
    for path in thermal_paths:
        try:
            raw = path.read_text(encoding="utf-8").strip()
            value = float(raw)
            return value / 1000.0 if value > 1000 else value
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return None


def read_cpu_freq_mhz() -> float | None:
    if psutil is None:
        return None
    try:
        freq = psutil.cpu_freq()
    except Exception:
        return None
    if freq is None:
        return None
    return float(freq.current)


def take_process_snapshot() -> ProcessSnapshot:
    if psutil is None:
        return ProcessSnapshot(cpu_time_s=0.0, rss_bytes=None, vms_bytes=None, cpu_temp_c=read_cpu_temp_c(), cpu_freq_mhz=read_cpu_freq_mhz())
    process = psutil.Process(os.getpid())
    try:
        cpu_times = process.cpu_times()
        cpu_time_s = float(cpu_times.user + cpu_times.system)
    except Exception:
        cpu_time_s = 0.0
    try:
        memory_info = process.memory_info()
        rss_bytes = int(memory_info.rss)
        vms_bytes = int(memory_info.vms)
    except Exception:
        rss_bytes = None
        vms_bytes = None
    return ProcessSnapshot(
        cpu_time_s=cpu_time_s,
        rss_bytes=rss_bytes,
        vms_bytes=vms_bytes,
        cpu_temp_c=read_cpu_temp_c(),
        cpu_freq_mhz=read_cpu_freq_mhz(),
    )


def norm_text(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def tok(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = tok(pred)
    gold_tokens = tok(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    matched = 0
    gold_used = [False] * len(gold_tokens)
    for token in pred_tokens:
        for idx, gold_token in enumerate(gold_tokens):
            if not gold_used[idx] and token == gold_token:
                matched += 1
                gold_used[idx] = True
                break
    precision = matched / len(pred_tokens)
    recall = matched / len(gold_tokens)
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def rouge_l_f1(pred: str, gold: str) -> float:
    pred_tokens = tok(pred)
    gold_tokens = tok(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    dp = [[0] * (len(gold_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i in range(1, len(pred_tokens) + 1):
        for j in range(1, len(gold_tokens) + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if norm_text(pred) == norm_text(gold) else 0.0


def seq_sim(pred: str, gold: str) -> float:
    return SequenceMatcher(None, norm_text(pred), norm_text(gold)).ratio()


def extract_label(text: str | None) -> str | None:
    if text is None:
        return None
    lowered = str(text).lower()
    matches = re.findall(r"answer\s*:\s*([a-z_\-]+)", lowered)
    if matches:
        candidate = matches[-1]
        return candidate if candidate in LABELS else None
    positions = [(lowered.rfind(label), label) for label in LABELS if lowered.rfind(label) != -1]
    if positions:
        positions.sort()
        return positions[-1][1]
    return None


def build_text_metrics(predictions: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    scores = {"token_f1": [], "rouge_l_f1": [], "exact_match": [], "seq_sim": []}
    detailed_rows: list[dict[str, Any]] = []

    y_true: list[str] = []
    y_pred: list[str] = []
    for row in predictions:
        prediction = str(row["prediction"])
        target = str(row["target"])
        sample_scores = {
            "token_f1": token_f1(prediction, target),
            "rouge_l_f1": rouge_l_f1(prediction, target),
            "exact_match": exact_match(prediction, target),
            "seq_sim": seq_sim(prediction, target),
        }
        for key, value in sample_scores.items():
            scores[key].append(value)

        target_label = extract_label(target)
        pred_label = extract_label(prediction)
        if target_label is not None and pred_label is not None:
            y_true.append(target_label)
            y_pred.append(pred_label)

        detailed_rows.append(
            {
                **row,
                **sample_scores,
                "target_label": target_label,
                "predicted_label": pred_label,
            }
        )

    total = len(y_true)
    correct = sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred)
    accuracy = (correct / total) if total else 0.0

    macro_f1 = 0.0
    for label in LABELS:
        tp = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(y_true, y_pred) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        macro_f1 += 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    macro_f1 = macro_f1 / len(LABELS) if LABELS else 0.0

    summary = {
        "n_predictions": len(predictions),
        "token_f1_mean": float(np.mean(scores["token_f1"])) if scores["token_f1"] else 0.0,
        "rouge_l_f1_mean": float(np.mean(scores["rouge_l_f1"])) if scores["rouge_l_f1"] else 0.0,
        "exact_match_mean": float(np.mean(scores["exact_match"])) if scores["exact_match"] else 0.0,
        "seq_sim_mean": float(np.mean(scores["seq_sim"])) if scores["seq_sim"] else 0.0,
        "classification_support": total,
        "classification_correct": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }
    return summary, detailed_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenTSLM Flamingo inference and collect system metrics.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "obd_cot_gpt5.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "data" / "obd_flamingo_best.pt")
    parser.add_argument("--llm-id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--llm-dtype", choices=["float32", "bfloat16", "float16"], default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics-interval", type=float, default=0.2)
    parser.add_argument("--cross-attn-every-n-layers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "raspberry_pi_inference")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--keep-congested", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Set HF_HUB_OFFLINE=1 before loading models.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "dependency"
        raise RuntimeError(
            f"Missing dependency '{missing_name}' required for Flamingo inference. "
            "Install the OpenFlamingo extras before running this script."
        ) from exc

    set_seed(args.seed)
    device = detect_device(args.device)
    llm_dtype = resolve_dtype(device, args.llm_dtype)
    rows = load_rows(args.dataset, drop_congested=not args.keep_congested)
    split_rows_selected = split_rows(rows, seed=args.seed, split=args.split)
    if args.limit is not None:
        split_rows_selected = split_rows_selected[: args.limit]
    samples = [prepare_sample(row) for row in split_rows_selected]

    if not samples:
        raise RuntimeError("No samples available for inference with the selected arguments.")

    run_name = args.run_name or f"obd_flamingo_{args.llm_id.split('/')[-1].replace('-', '_').replace('.', '_').lower()}_{args.split}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / f"{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.llm_id} on device={device} dtype={llm_dtype}")
    model = OpenTSLMFlamingo(
        device=device,
        llm_id=args.llm_id,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        llm_torch_dtype=llm_dtype,
        llm_low_cpu_mem_usage=True,
        use_device_map=False,
    ).to(device)
    model.load_from_file(str(args.checkpoint))
    model.eval()

    collector = SystemMetricsCollector(disk_path=PROJECT_ROOT)
    monitor = SystemMetricsMonitor(
        label=run_name,
        interval_s=args.metrics_interval,
        disk_path=PROJECT_ROOT,
        metadata={
            "script": "run_obd_flamingo_inference.py",
            "model_type": "flamingo",
            "llm_id": args.llm_id,
            "device": device,
            "dataset": str(args.dataset),
            "checkpoint": str(args.checkpoint),
            "split": args.split,
        },
    ).start()
    monitor.mark("setup", n_samples=len(samples), max_new_tokens=args.max_new_tokens)

    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for step, sample in enumerate(samples, start=1):
            batch = [sample]
            before = take_process_snapshot()
            start_snapshot = collector.snapshot(label=run_name, phase="before_step", step=step, sample_id=sample["id"])
            started_at = time.perf_counter()
            generated = model.generate(batch, max_new_tokens=args.max_new_tokens)[0].strip()
            wall_time_s = time.perf_counter() - started_at
            after = take_process_snapshot()
            end_snapshot = collector.snapshot(label=run_name, phase="after_step", step=step, sample_id=sample["id"])

            process_cpu_time_delta_s = max(0.0, after.cpu_time_s - before.cpu_time_s)
            process_cpu_percent_avg = (100.0 * process_cpu_time_delta_s / wall_time_s) if wall_time_s > 0 else 0.0
            rss_delta_bytes = None if before.rss_bytes is None or after.rss_bytes is None else after.rss_bytes - before.rss_bytes

            prediction_row = {
                "id": sample["id"],
                "prediction": generated,
                "target": sample["answer"],
                "prompt": {
                    "pre_prompt": sample["pre_prompt"],
                    "time_series_text": sample["time_series_text"],
                    "post_prompt": sample["post_prompt"],
                },
                "resource_metrics": {
                    "inference_latency_s": wall_time_s,
                    "process_cpu_time_delta_s": process_cpu_time_delta_s,
                    "process_cpu_percent_avg": process_cpu_percent_avg,
                    "process_rss_before_bytes": before.rss_bytes,
                    "process_rss_after_bytes": after.rss_bytes,
                    "process_rss_delta_bytes": rss_delta_bytes,
                    "process_vms_after_bytes": after.vms_bytes,
                    "cpu_temp_before_c": before.cpu_temp_c,
                    "cpu_temp_after_c": after.cpu_temp_c,
                    "cpu_freq_before_mhz": before.cpu_freq_mhz,
                    "cpu_freq_after_mhz": after.cpu_freq_mhz,
                    "system_cpu_after_percent": end_snapshot.get("system_cpu_percent"),
                    "system_ram_after_percent": end_snapshot.get("system_ram_percent"),
                    "system_ram_used_bytes": end_snapshot.get("system_ram_used_bytes"),
                },
            }
            predictions.append(prediction_row)

            monitor.mark(
                "inference_step",
                step=step,
                sample_id=sample["id"],
                inference_latency_s=wall_time_s,
                process_cpu_time_delta_s=process_cpu_time_delta_s,
                process_cpu_percent_avg=process_cpu_percent_avg,
                process_rss_before_bytes=before.rss_bytes,
                process_rss_after_bytes=after.rss_bytes,
                process_rss_delta_bytes=rss_delta_bytes,
                process_vms_after_bytes=after.vms_bytes,
                cpu_temp_before_c=before.cpu_temp_c,
                cpu_temp_after_c=after.cpu_temp_c,
                cpu_freq_before_mhz=before.cpu_freq_mhz,
                cpu_freq_after_mhz=after.cpu_freq_mhz,
                system_cpu_before_percent=start_snapshot.get("system_cpu_percent"),
                system_cpu_after_percent=end_snapshot.get("system_cpu_percent"),
                system_ram_after_percent=end_snapshot.get("system_ram_percent"),
                system_ram_used_bytes=end_snapshot.get("system_ram_used_bytes"),
                prediction_chars=len(generated),
            )
            print(f"[{step}/{len(samples)}] {sample['id']} latency={wall_time_s:.3f}s cpu={process_cpu_percent_avg:.1f}%")

    final_row = monitor.stop(final_phase="finished", n_predictions=len(predictions))
    text_summary, detailed_rows = build_text_metrics(predictions)
    system_summary = monitor.summary().to_dict()

    predictions_path = output_dir / "predictions.jsonl"
    detailed_path = output_dir / "detailed_metrics.jsonl"
    system_csv_path = output_dir / "system_metrics.csv"
    system_summary_path = output_dir / "system_summary.json"
    final_summary_path = output_dir / "summary.json"

    write_jsonl(predictions_path, predictions)
    write_jsonl(detailed_path, detailed_rows)
    monitor.to_csv(system_csv_path)
    system_summary_path.write_text(json.dumps(system_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final_summary = {
        "run_name": run_name,
        "timestamp_utc": timestamp,
        "script": "run_obd_flamingo_inference.py",
        "model_type": "flamingo",
        "llm_id": args.llm_id,
        "device": device,
        "llm_dtype": str(llm_dtype),
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "seed": args.seed,
        "n_samples": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "final_monitor_row": final_row,
        "text_metrics": text_summary,
        "system_metrics_summary": system_summary,
        "artifacts": {
            "predictions_jsonl": str(predictions_path),
            "detailed_metrics_jsonl": str(detailed_path),
            "system_metrics_csv": str(system_csv_path),
            "system_summary_json": str(system_summary_path),
        },
    }
    final_summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved predictions to {predictions_path}")
    print(f"Saved detailed metrics to {detailed_path}")
    print(f"Saved system metrics CSV to {system_csv_path}")
    print(f"Saved run summary to {final_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
