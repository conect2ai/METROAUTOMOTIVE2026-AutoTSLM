#!/usr/bin/env python3
import argparse
import base64
import json
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from generate_obd_alignment_dataset import (
    FEATURE_COLUMNS,
    CORE_PLOT_FEATURES,
    build_windows,
)

LABELS = ["economical", "normal", "aggressive", "congested"]
DISSIMILAR = {
    "economical": "aggressive",
    "aggressive": "economical",
    "congested": "aggressive",
    "normal": "aggressive",
}


def classify_style(sample) -> str:
    speed = float(np.mean(sample.values[FEATURE_COLUMNS.index("Speed")]))
    rpm = float(np.mean(sample.values[FEATURE_COLUMNS.index("RPM")]))
    load = float(np.mean(sample.values[FEATURE_COLUMNS.index("EngineLoad")]))
    throttle = float(np.mean(sample.values[FEATURE_COLUMNS.index("ThrottlePos")]))

    accel_x = sample.values[FEATURE_COLUMNS.index("Accelerometer_X")]
    accel_y = sample.values[FEATURE_COLUMNS.index("Accelerometer_Y")]
    accel_z = sample.values[FEATURE_COLUMNS.index("Accelerometer_Z")]
    accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    accel_std = float(np.std(accel_mag))

    stop_ratio = float(np.mean(sample.values[FEATURE_COLUMNS.index("Speed")] < 2.0))

    if stop_ratio > 0.5 or speed < 5:
        return "congested"
    if rpm > 2800 or throttle > 60 or load > 70 or accel_std > 0.6:
        return "aggressive"
    if 10 <= speed <= 60 and rpm < 1900 and throttle < 35:
        return "economical"
    return "normal"


def build_prompt(correct_label: str, dissimilar_label: str) -> str:
    return (
        "You are shown a time-series plot of vehicle telemetry over a 120-sample window.\n"
        "This data corresponds to one of two possible activities:\n"
        f"[{correct_label}]\n"
        f"[{dissimilar_label}]\n"
        "Your task is to classify the activity based on analysis of the data.\n"
        "Instructions:\n"
        "- Begin by analyzing the time-series without assuming a specific label.\n"
        "- Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.\n"
        "- Write your rationale as a single, natural paragraph, do not use bullet points, numbered steps, or section headings.\n"
        "- Do not refer back to the plot or to the act of visual analysis in your rationale; the plot is only for reference but you should reason about the time-series data.\n"
        "- Do **not** assume any answer at the beginning, analyze as if you do not yet know which class is correct.\n"
        "- Do **not** mention either class label until the final sentence.\n"
        "- Make sure that your last word is the answer. You MUST end your response with \"Answer: "
        f"{correct_label}\".\n"
    )


def render_plot(sample, out_path: Path) -> None:
    ts = sample.values  # [features, time]
    t = np.arange(ts.shape[1])

    speed = ts[FEATURE_COLUMNS.index("Speed")]
    rpm = ts[FEATURE_COLUMNS.index("RPM")]
    load = ts[FEATURE_COLUMNS.index("EngineLoad")]
    throttle = ts[FEATURE_COLUMNS.index("ThrottlePos")]

    fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t, speed)
    axes[0].set_ylabel("Speed (km/h)")
    axes[1].plot(t, rpm)
    axes[1].set_ylabel("RPM")
    axes[2].plot(t, load)
    axes[2].set_ylabel("Engine Load (%)")
    axes[3].plot(t, throttle)
    axes[3].set_ylabel("Throttle (%)")
    axes[3].set_xlabel("t")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def encode_image_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def call_openai(prompt: str, image_path: Path, model: str, temperature: float) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package not installed. Install with: pip install openai") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    b64 = encode_image_base64(image_path)
    data_url = f"data:image/png;base64,{b64}"

    kwargs = dict(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    # gpt-5 does not accept temperature; only set when supported
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = temperature

    resp = client.responses.create(**kwargs)

    # Prefer output_text if available; fallback to structured output
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()

    # Fallback: concatenate text parts
    parts = []
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    parts.append(c.text)
    return "\n".join(parts).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CoT rationales from OBD-II windows using GPT-4o.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-jsonl", default="data/obd_cot_gpt4o.jsonl")
    parser.add_argument("--plot-dir", default="data/obd_cot_plots")
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument("--samples-per-file", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--openai-model", default="gpt-5")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    csv_files = sorted(input_dir.rglob("*.csv"))
    if args.max_files and args.max_files > 0:
        csv_files = csv_files[: args.max_files]
    if not args.allow_missing:
        # heterogeneous OBD exports often miss columns; default to allow_missing
        args.allow_missing = True

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output_jsonl)
    rows = []
    total = 0

    for csv_path in csv_files:
        # reuse existing loader
        import pandas as pd
        df = pd.read_csv(csv_path, low_memory=False)

        windows = build_windows(
            df,
            n_samples=args.samples_per_file if args.samples_per_file > 0 else None,
            window_size=args.window_size,
            stride=args.stride,
            source_label=str(csv_path),
            allow_missing=args.allow_missing,
            require_features=CORE_PLOT_FEATURES,
        )

        for sample in windows:
            if args.max_samples and total >= args.max_samples:
                break
            total += 1

            label = classify_style(sample)
            dissimilar = DISSIMILAR.get(label, "agressivo")
            prompt = build_prompt(label, dissimilar)

            plot_path = plot_dir / f"obd_cot_{total:06d}.png"
            render_plot(sample, plot_path)

            rationale = None
            status = "pending"
            if args.use_openai:
                try:
                    rationale = call_openai(prompt, plot_path, args.openai_model, args.temperature)
                    status = "ok"
                except Exception as exc:
                    rationale = None
                    status = f"error:{exc}".strip()

            row = {
                "id": f"obd_cot_{total:06d}",
                "source_file": str(csv_path),
                "window_start": sample.start_idx,
                "window_end": sample.end_idx,
                "plot_path": str(plot_path),
                "label": label,
                "dissimilar_label": dissimilar,
                "prompt": prompt,
                "rationale": rationale,
                "generation_status": status,
                # training-compatible fields
                "pre_prompt": prompt,
                "time_series_text": ["Sinais OBD-II (janela de 120 amostras)."],
                "post_prompt": "Rationale:",
                "answer": rationale or "",
                "time_series": sample.values.tolist(),
            }
            rows.append(row)

        if args.max_samples and total >= args.max_samples:
            break

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
