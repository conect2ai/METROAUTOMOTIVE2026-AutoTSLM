"""Utilities for sampling and logging system metrics during experiments."""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, Optional

try:
    import psutil  # type: ignore

    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False

try:
    import pynvml  # type: ignore

    _NVML_AVAILABLE = True
except ImportError:
    pynvml = None
    _NVML_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _flatten(prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in payload.items()}


@dataclass
class MetricsSummary:
    label: str
    sample_count: int
    started_at: str
    finished_at: str
    total_duration_s: float
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "sample_count": self.sample_count,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_duration_s": self.total_duration_s,
            **self.metrics,
        }


class SystemMetricsCollector:
    """Collect point-in-time system and process metrics."""

    def __init__(self, disk_path: str | os.PathLike[str] | None = None):
        self.disk_path = str(Path(disk_path or os.getcwd()).resolve())
        self.process = psutil.Process(os.getpid()) if _PSUTIL_AVAILABLE else None
        self._nvml_initialized = False
        self._nvml_handles: list[Any] = []

        if self.process is not None:
            try:
                self.process.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def _ensure_nvml(self) -> bool:
        if not _NVML_AVAILABLE:
            return False
        if self._nvml_initialized:
            return True
        try:
            pynvml.nvmlInit()
            self._nvml_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
            self._nvml_initialized = True
            return True
        except Exception:
            self._nvml_handles = []
            self._nvml_initialized = False
            return False

    def _collect_cpu_and_memory(self) -> Dict[str, Any]:
        if self.process is None or not _PSUTIL_AVAILABLE:
            return {}

        payload: Dict[str, Any] = {}
        try:
            vm = psutil.virtual_memory()
            payload.update(
                {
                    "system_cpu_percent": _safe_float(psutil.cpu_percent(interval=None)),
                    "system_ram_total_bytes": _safe_int(vm.total),
                    "system_ram_available_bytes": _safe_int(vm.available),
                    "system_ram_used_bytes": _safe_int(vm.used),
                    "system_ram_percent": _safe_float(vm.percent),
                }
            )
        except Exception:
            pass

        try:
            mem = self.process.memory_info()
            payload.update(
                {
                    "process_cpu_percent": _safe_float(
                        self.process.cpu_percent(interval=None)
                    ),
                    "process_rss_bytes": _safe_int(mem.rss),
                    "process_vms_bytes": _safe_int(mem.vms),
                }
            )
        except Exception:
            pass

        try:
            io = self.process.io_counters()
            payload.update(
                {
                    "process_read_count": _safe_int(getattr(io, "read_count", None)),
                    "process_write_count": _safe_int(getattr(io, "write_count", None)),
                    "process_read_bytes": _safe_int(getattr(io, "read_bytes", None)),
                    "process_write_bytes": _safe_int(getattr(io, "write_bytes", None)),
                }
            )
        except Exception:
            pass

        try:
            disk = psutil.disk_usage(self.disk_path)
            payload.update(
                {
                    "disk_total_bytes": _safe_int(disk.total),
                    "disk_used_bytes": _safe_int(disk.used),
                    "disk_free_bytes": _safe_int(disk.free),
                    "disk_percent": _safe_float(disk.percent),
                }
            )
        except Exception:
            pass

        return payload

    def _collect_torch_cuda(self) -> Dict[str, Any]:
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}

        payload: Dict[str, Any] = {"cuda_device_count": torch.cuda.device_count()}
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        for device_idx in range(torch.cuda.device_count()):
            try:
                payload.update(
                    _flatten(
                        f"cuda_{device_idx}",
                        {
                            "memory_allocated_bytes": _safe_int(
                                torch.cuda.memory_allocated(device_idx)
                            ),
                            "memory_reserved_bytes": _safe_int(
                                torch.cuda.memory_reserved(device_idx)
                            ),
                            "max_memory_allocated_bytes": _safe_int(
                                torch.cuda.max_memory_allocated(device_idx)
                            ),
                            "max_memory_reserved_bytes": _safe_int(
                                torch.cuda.max_memory_reserved(device_idx)
                            ),
                        },
                    )
                )
            except Exception:
                continue

        return payload

    def _collect_nvml(self) -> Dict[str, Any]:
        if not self._ensure_nvml():
            return {}

        payload: Dict[str, Any] = {}
        current_pid = os.getpid()
        for device_idx, handle in enumerate(self._nvml_handles):
            gpu_payload: Dict[str, Any] = {}
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_payload["utilization_percent"] = _safe_float(utilization.gpu)
                gpu_payload["memory_utilization_percent"] = _safe_float(utilization.memory)
            except Exception:
                pass

            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_payload["memory_total_bytes"] = _safe_int(mem.total)
                gpu_payload["memory_used_bytes"] = _safe_int(mem.used)
                gpu_payload["memory_free_bytes"] = _safe_int(mem.free)
            except Exception:
                pass

            try:
                gpu_payload["temperature_c"] = _safe_float(
                    pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception:
                pass

            try:
                gpu_payload["power_draw_w"] = _safe_float(
                    pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                )
            except Exception:
                pass

            process_memory = 0
            found_process = False
            for getter_name in (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetGraphicsRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ):
                getter = getattr(pynvml, getter_name, None)
                if getter is None:
                    continue
                try:
                    for proc in getter(handle):
                        if int(getattr(proc, "pid", -1)) != current_pid:
                            continue
                        used = getattr(proc, "usedGpuMemory", None)
                        if used is None or used < 0:
                            continue
                        process_memory += int(used)
                        found_process = True
                except Exception:
                    continue
            if found_process:
                gpu_payload["process_memory_bytes"] = process_memory

            payload.update(_flatten(f"gpu_{device_idx}", gpu_payload))

        return payload

    def snapshot(
        self,
        label: str = "",
        phase: str = "",
        step: int | None = None,
        **metadata: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "timestamp_utc": _utc_now().isoformat(),
            "label": label,
            "phase": phase,
            "step": step,
        }
        payload.update(self._collect_cpu_and_memory())
        payload.update(self._collect_torch_cuda())
        payload.update(self._collect_nvml())
        payload.update(metadata)
        return payload

    def reset_torch_peak_memory(self) -> None:
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        for device_idx in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(device_idx)
            except Exception:
                continue


class SystemMetricsMonitor:
    """Continuously sample system metrics in the background."""

    def __init__(
        self,
        label: str,
        interval_s: float = 1.0,
        disk_path: str | os.PathLike[str] | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.label = label
        self.interval_s = interval_s
        self.collector = SystemMetricsCollector(disk_path=disk_path)
        self.metadata = metadata or {}
        self.rows: list[Dict[str, Any]] = []
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._step = 0

    def start(self, reset_cuda_peaks: bool = True) -> "SystemMetricsMonitor":
        self.rows = []
        self.started_at = _utc_now()
        self.finished_at = None
        self._step = 0
        self._stop_event.clear()
        if reset_cuda_peaks:
            self.collector.reset_torch_peak_memory()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self.mark(phase="sample")

    def mark(
        self,
        phase: str,
        step: int | None = None,
        **metadata: Any,
    ) -> Dict[str, Any]:
        effective_step = self._step if step is None else step
        row = self.collector.snapshot(
            label=self.label,
            phase=phase,
            step=effective_step,
            **self.metadata,
            **metadata,
        )
        if self.started_at is not None:
            row["elapsed_s"] = (_utc_now() - self.started_at).total_seconds()
        self.rows.append(row)
        self._step = effective_step + 1
        return row

    def stop(self, final_phase: str = "finished", **metadata: Any) -> Dict[str, Any]:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=max(self.interval_s * 2.0, 1.0))
            self._thread = None
        self.finished_at = _utc_now()
        return self.mark(phase=final_phase, **metadata)

    def __enter__(self) -> "SystemMetricsMonitor":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop(
            final_phase="error" if exc else "finished",
            error=str(exc) if exc else "",
        )

    def to_csv(self, path: str | os.PathLike[str]) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in self.rows for key in row.keys()})
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return output_path

    def to_jsonl(self, path: str | os.PathLike[str]) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        return output_path

    def summary(self) -> MetricsSummary:
        started_at = self.started_at or _utc_now()
        finished_at = self.finished_at or _utc_now()
        duration_s = (finished_at - started_at).total_seconds()

        numeric_keys = {
            key
            for row in self.rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in {"step"}
        }
        metrics: Dict[str, Any] = {}
        for key in sorted(numeric_keys):
            values = [
                float(row[key])
                for row in self.rows
                if isinstance(row.get(key), (int, float))
            ]
            if not values:
                continue
            metrics[f"{key}_max"] = max(values)
            metrics[f"{key}_mean"] = mean(values)

        return MetricsSummary(
            label=self.label,
            sample_count=len(self.rows),
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            total_duration_s=duration_s,
            metrics=metrics,
        )


def measure_function(
    label: str,
    fn: Callable[..., Any],
    *args: Any,
    interval_s: float = 0.5,
    disk_path: str | os.PathLike[str] | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run a function while collecting system metrics."""

    monitor = SystemMetricsMonitor(
        label=label,
        interval_s=interval_s,
        disk_path=disk_path,
        metadata=metadata,
    ).start()
    monitor.mark(phase="start")
    started = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        total_time = time.perf_counter() - started
        monitor.stop(final_phase="finished", total_duration_measured_s=total_time)

    return {
        "result": result,
        "summary": monitor.summary().to_dict(),
        "rows": monitor.rows,
    }


def append_summary_csv(
    path: str | os.PathLike[str],
    summaries: Iterable[Dict[str, Any]],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(summaries)
    if not rows:
        return output_path

    fieldnames = sorted({key for row in rows for key in row.keys()})
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    return output_path
