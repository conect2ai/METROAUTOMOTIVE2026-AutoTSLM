#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Speed",
    "RPM",
    "EngineLoad",
    "ThrottlePos",
    "CoolantTemp",
    "Maf",
    "FuelLevel",
    "FuelPressure",
    "IntakeMAP",
    "TimingAdvance",
    "EngineFuelRate",
    "AirFuelRatio",
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
]

CORE_PLOT_FEATURES = ["Speed", "RPM", "EngineLoad", "ThrottlePos"]
ALT_COLUMN_MAP = {
    "Speed": [
        "Speed (OBD)(km/h)",
        "ECU(7E9): Speed (OBD)(km/h)",
        "Speed (GPS)(km/h)",
        "GPS Speed (Meters/second)",
        "speed",
    ],
    "RPM": [
        "Engine RPM(rpm)",
        "ECU(7E9): Engine RPM(rpm)",
        "rpm",
    ],
    "EngineLoad": [
        "Engine Load(%)",
        "ECU(7E9): Engine Load(%)",
        "engine_load",
    ],
    "ThrottlePos": [
        "Throttle Position(Manifold)(%)",
        "ECU(7E9): Throttle Position(Manifold)(%)",
        "Relative Throttle Position(%)",
        "Absolute Throttle Position B(%)",
        "throttle",
    ],
    "CoolantTemp": [
        "Engine Coolant Temperature(°C)",
        "ECU(7E9): Engine Coolant Temperature(°C)",
        "coolant_temp",
    ],
    "Maf": [
        "Mass Air Flow Rate(g/s)",
        "maf",
    ],
    "FuelLevel": [
        "Fuel Level (From Engine ECU)(%)",
        "fuel_level",
    ],
    "FuelPressure": [
        "Fuel Rail Pressure(kpa)",
    ],
    "IntakeMAP": [
        "Intake Manifold Pressure(kpa)",
    ],
    "TimingAdvance": [
        "Timing Advance(°)",
        "timing_advance",
    ],
    "EngineFuelRate": [
        "Fuel flow rate/hour(l/hr)",
        "Fuel flow rate/minute(cc/min)",
    ],
    "AirFuelRatio": [
        "Air Fuel Ratio(Measured)(:1)",
        "Air Fuel Ratio(Commanded)(:1)",
        "Commanded Equivalence Ratio(lambda)",
    ],
    "Accelerometer_X": [
        "Acceleration Sensor(X axis)(g)",
        "G(x)",
        "accel_x",
    ],
    "Accelerometer_Y": [
        "Acceleration Sensor(Y axis)(g)",
        "G(y)",
        "accel_y",
    ],
    "Accelerometer_Z": [
        "Acceleration Sensor(Z axis)(g)",
        "G(z)",
        "accel_z",
    ],
}

TIME_COLUMNS = ["TimeSensor", "Device Time", "GPS Time", "timestamp", "ts"]

PROMPT_FEATURES = [
    "Speed",
    "RPM",
    "EngineLoad",
    "ThrottlePos",
    "FuelLevel",
    "EngineFuelRate",
]


@dataclass
class WindowSample:
    source_file: str
    start_idx: int
    end_idx: int
    timestamps: List[str]
    values: np.ndarray  # [features, time]


def choose_group_column(df: pd.DataFrame) -> str:
    for col in ["TripFolder", "SourceFile", "SessionDate", "Device"]:
        if col in df.columns:
            return col
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def extract_feature_frame(
    df: pd.DataFrame,
    allow_missing: bool,
    require_features: Optional[list[str]] = None,
) -> pd.DataFrame:
    df = normalize_columns(df)
    out = {}
    require_features = set(require_features or [])
    for feature in FEATURE_COLUMNS:
        candidates = ALT_COLUMN_MAP.get(feature, [])
        found = None
        for col in candidates:
            if col in df.columns:
                found = col
                break
        if found is None:
            if feature in require_features:
                raise ValueError(f"Missing required feature {feature} (candidates: {candidates})")
            if allow_missing:
                out[feature] = pd.Series([np.nan] * len(df))
                continue
            raise ValueError(f"Missing feature {feature} (candidates: {candidates})")

        series = pd.to_numeric(df[found], errors="coerce")
        # unit conversions
        if feature == "Speed" and found == "GPS Speed (Meters/second)":
            series = series * 3.6
        if feature == "EngineFuelRate" and found == "Fuel flow rate/minute(cc/min)":
            # convert cc/min -> L/h
            series = series * 0.06

        out[feature] = series

    return pd.DataFrame(out)


def extract_time_column(df: pd.DataFrame) -> Optional[pd.Series]:
    df = normalize_columns(df)
    for col in TIME_COLUMNS:
        if col in df.columns:
            return df[col].astype(str)
    return None


def build_windows(
    df: pd.DataFrame,
    n_samples: Optional[int],
    window_size: int,
    stride: int,
    source_label: Optional[str] = None,
    allow_missing: bool = False,
    require_features: Optional[list[str]] = None,
) -> List[WindowSample]:
    group_col = choose_group_column(df)
    time_col = extract_time_column(df)
    windows: List[WindowSample] = []

    grouped = df.groupby(group_col) if group_col is not None else [(None, df)]
    for group_name, g in grouped:
        g = g.sort_values("TimeSensor") if "TimeSensor" in g.columns else g
        g = g.reset_index(drop=True)

        feature_frame = extract_feature_frame(
            g,
            allow_missing=allow_missing,
            require_features=require_features,
        )
        feature_frame = feature_frame.interpolate(limit_direction="both")
        feature_frame = feature_frame.fillna(0.0)

        if len(feature_frame) < window_size:
            continue

        for start in range(0, len(feature_frame) - window_size + 1, stride):
            end = start + window_size
            window = feature_frame.iloc[start:end].to_numpy(dtype=np.float32).T
            if time_col is not None:
                timestamps = time_col.iloc[start:end].tolist()
            else:
                timestamps = [str(i) for i in range(start, end)]
            source_file = (
                source_label
                if source_label is not None
                else (
                    str(g.iloc[0]["SourceFile"])
                    if "SourceFile" in g.columns
                    else str(group_name)
                )
            )

            windows.append(
                WindowSample(
                    source_file=source_file,
                    start_idx=start,
                    end_idx=end - 1,
                    timestamps=timestamps,
                    values=window,
                )
            )
            if n_samples is not None and len(windows) >= n_samples:
                return windows

    return windows


def summarize_window(sample: WindowSample) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for i, feature in enumerate(FEATURE_COLUMNS):
        x = sample.values[i]
        stats[feature] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }
    return stats


def build_prompt(sample: WindowSample) -> str:
    stats = summarize_window(sample)
    compact_all = {
        k: {
            "mean": round(v["mean"], 2),
            "std": round(v["std"], 2),
            "min": round(v["min"], 2),
            "max": round(v["max"], 2),
        }
        for k, v in stats.items()
    }
    compact = {k: compact_all[k] for k in PROMPT_FEATURES}

    return (
        "Analyze an OBD-II window and return ONLY valid JSON with keys: caption, question, answer, risk_level, driving_style. "
        "Use English. caption short; question diagnostic; answer concise. "
        "risk_level must be low, medium, or high. "
        "driving_style must be one of: economical, normal, aggressive, congested.\n"
        f"Source: {sample.source_file}; interval={sample.start_idx}-{sample.end_idx}\n"
        f"Stats: {json.dumps(compact, ensure_ascii=False)}"
    )


def parse_json_from_text(text: str) -> Dict[str, str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.replace("```json", "").replace("```", "").strip()
    block = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = block.group(0) if block else text
    data = json.loads(candidate)
    return {
        "caption": str(data.get("caption", "")).strip(),
        "question": str(data.get("question", "")).strip(),
        "answer": str(data.get("answer", "")).strip(),
        "risk_level": str(data.get("risk_level", "medium")).strip().lower(),
        "driving_style": str(data.get("driving_style", "normal")).strip().lower(),
    }


def fallback_text(sample: WindowSample) -> Dict[str, str]:
    # Deterministic RNG per window for reproducible variety
    seed_str = f"{sample.source_file}:{sample.start_idx}-{sample.end_idx}"
    seed = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    speed = float(np.mean(sample.values[FEATURE_COLUMNS.index("Speed")]))
    rpm = float(np.mean(sample.values[FEATURE_COLUMNS.index("RPM")]))
    load = float(np.mean(sample.values[FEATURE_COLUMNS.index("EngineLoad")]))
    throttle = float(np.mean(sample.values[FEATURE_COLUMNS.index("ThrottlePos")]))

    speed_std = float(np.std(sample.values[FEATURE_COLUMNS.index("Speed")]))
    rpm_std = float(np.std(sample.values[FEATURE_COLUMNS.index("RPM")]))

    accel_x = sample.values[FEATURE_COLUMNS.index("Accelerometer_X")]
    accel_y = sample.values[FEATURE_COLUMNS.index("Accelerometer_Y")]
    accel_z = sample.values[FEATURE_COLUMNS.index("Accelerometer_Z")]
    accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    accel_std = float(np.std(accel_mag))

    stop_ratio = float(np.mean(sample.values[FEATURE_COLUMNS.index("Speed")] < 2.0))

    if stop_ratio > 0.5 or speed < 5:
        style = "congested"
    elif rpm > 2800 or throttle > 60 or load > 70:
        style = "aggressive"
    elif 10 <= speed <= 60 and rpm < 1900 and throttle < 35:
        style = "economical"
    else:
        style = "normal"

    if load > 85 or rpm > 3500 or accel_std > 0.8:
        risk = "high"
    elif rpm > 2500 or throttle > 50 or accel_std > 0.5:
        risk = "medium"
    else:
        risk = "low"

    q_templates = {
        "congested": [
            "Are there signs of slow traffic or frequent stops in this segment?",
            "Does this pattern suggest congested driving?",
            "Does the vehicle behavior indicate low traffic flow?",
            "Are stops frequent in this interval?",
            "Does the segment suggest heavy traffic with low speed variation?",
        ],
        "aggressive": [
            "Are there signs of aggressive driving or higher consumption in this segment?",
            "Do the signals suggest strong accelerations or sporty driving?",
            "Does the behavior indicate high engine demand?",
            "Are there indications of intense acceleration in this interval?",
            "Does the segment suggest a more aggressive driving style?",
        ],
        "economical": [
            "Does the observed pattern favor fuel efficiency?",
            "Are there signs of efficient driving in this segment?",
            "Does this interval suggest an economical driving style?",
            "Does the behavior indicate a focus on efficiency?",
            "Does the segment show smooth and efficient driving?",
        ],
        "normal": [
            "Does the segment indicate stable and regular driving?",
            "Are there signs of normal driving behavior?",
            "Is the vehicle behavior balanced?",
            "Does the segment suggest routine driving without extremes?",
            "Do the signals indicate regular vehicle operation?",
        ],
    }

    a_templates = {
        "congested": [
            "Low speed and low variation suggest slow traffic and frequent stops.",
            "The pattern is consistent with heavy traffic and low flow.",
            "Frequent stops indicate congestion in this segment.",
            "Low average speed with a high stop rate points to congestion.",
            "The interval shows behavior typical of saturated roads.",
        ],
        "aggressive": [
            "Higher RPM and acceleration indicate more aggressive driving and higher consumption.",
            "The combination of load and throttle suggests higher engine demand.",
            "Acceleration and RPM peaks signal sportier driving.",
            "The pattern indicates more intense engine use and higher consumption.",
            "Acceleration variability suggests more aggressive driving.",
        ],
        "economical": [
            "Moderate speed and low RPM suggest more economical driving.",
            "Low load and smooth acceleration indicate efficiency.",
            "The pattern is consistent with efficient and stable driving.",
            "Low variation and low RPM point to fuel efficiency.",
            "The segment shows moderate engine use and good consumption.",
        ],
        "normal": [
            "The signals indicate regular driving without relevant spikes.",
            "Moderate variations indicate normal behavior.",
            "The segment looks stable with moderate engine demand.",
            "The pattern is consistent with everyday driving.",
            "There are no strong signs of aggressive driving or congestion.",
        ],
    }

    question = rng.choice(q_templates[style])
    answer = rng.choice(a_templates[style])

    caption = (
        f"Window with average speed {speed:.1f} km/h, "
        f"average RPM {rpm:.0f}, load {load:.0f}%, "
        f"throttle {throttle:.0f}% and stop ratio {stop_ratio:.0%}."
    )

    return {
        "caption": caption,
        "question": question,
        "answer": answer,
        "risk_level": risk,
        "driving_style": style,
    }


def call_ollama(prompt: str, model: str, host: str, timeout: int) -> str:
    cmd = [
        "ollama",
        "run",
        model,
        prompt,
    ]
    env = {**os.environ, "OLLAMA_HOST": host}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ollama run failed")
    return proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small OBD-II text alignment dataset using Ollama.")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-jsonl", default="data/obd_alignment_small.jsonl")
    parser.add_argument("--output-csv", default="data/obd_alignment_small.csv")
    parser.add_argument("--samples", type=int, default=24, help="Total samples when using a single CSV.")
    parser.add_argument("--samples-per-file", type=int, default=4, help="Samples per CSV when using --input-dir.")
    parser.add_argument("--max-samples", type=int, default=0, help="Global cap; 0 means no cap.")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--host", default="127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--use-ollama", action="store_true", help="If set, call Ollama for text generation.")
    parser.add_argument("--allow-missing", action="store_true", help="If set, missing features are filled with zeros.")
    args = parser.parse_args()

    if args.input_csv is None and args.input_dir is None:
        args.input_csv = "descarbonize_dataset.csv"

    out_jsonl = Path(args.output_jsonl)
    out_csv = Path(args.output_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    csv_files: List[Path] = []
    if args.input_dir:
        csv_files = sorted(Path(args.input_dir).rglob("*.csv"))
        if not csv_files:
            raise RuntimeError(f"No CSV files found in {args.input_dir}")
        # default to allow missing features for heterogeneous datasets
        if not args.allow_missing:
            args.allow_missing = True
    else:
        csv_files = [Path(args.input_csv)]

    rows = []
    total_written = 0
    for file_idx, csv_path in enumerate(csv_files, start=1):
        df = pd.read_csv(csv_path, low_memory=False)

        per_file = args.samples_per_file if args.input_dir else args.samples
        if per_file <= 0:
            per_file = None

        try:
            windows = build_windows(
                df,
                n_samples=per_file,
                window_size=args.window_size,
                stride=args.stride,
                source_label=str(csv_path),
                allow_missing=args.allow_missing,
            )
        except ValueError as exc:
            print(f"Skipping {csv_path} ({exc})", flush=True)
            continue
        if not windows:
            print(f"Skipping {csv_path} (no windows)", flush=True)
            continue

        for sample in windows:
            if args.max_samples and total_written >= args.max_samples:
                break

            total_written += 1
            print(
                f"[{total_written}] generating text for {sample.source_file} ({sample.start_idx}-{sample.end_idx})",
                flush=True,
            )
            prompt = build_prompt(sample)
            if args.use_ollama:
                try:
                    raw = call_ollama(prompt=prompt, model=args.model, host=args.host, timeout=args.timeout)
                    text_data = parse_json_from_text(raw)
                    generation_status = "ok"
                except Exception:
                    text_data = fallback_text(sample)
                    generation_status = "fallback"
            else:
                text_data = fallback_text(sample)
                generation_status = "template"

            row = {
                "id": f"obd_{total_written:06d}",
            "source_file": sample.source_file,
            "window_start": sample.start_idx,
            "window_end": sample.end_idx,
            "start_time": sample.timestamps[0],
            "end_time": sample.timestamps[-1],
            "feature_names": FEATURE_COLUMNS,
            "time_series": sample.values.tolist(),
            "pre_prompt": "Analyze the vehicle time-series signals in the segment below.",
            "time_series_text": [text_data["caption"]],
            "post_prompt": text_data["question"],
            "answer": text_data["answer"],
            "risk_level": text_data["risk_level"],
            "driving_style": text_data["driving_style"],
            "llm_model": args.model,
            "generation_status": generation_status,
            }
            rows.append(row)

        if args.max_samples and total_written >= args.max_samples:
            break

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                "id": row["id"],
                "source_file": row["source_file"],
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "caption": row["time_series_text"][0],
                "question": row["post_prompt"],
                "answer": row["answer"],
                "risk_level": row["risk_level"],
                "driving_style": row["driving_style"],
                "llm_model": row["llm_model"],
                "generation_status": row["generation_status"],
            }
        )

    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} samples to {out_jsonl} and {out_csv}")


if __name__ == "__main__":
    main()
