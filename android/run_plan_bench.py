#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.io import wavfile
import torch


@dataclasses.dataclass(frozen=True)
class Case:
    flavor: str
    model: str
    sample: str
    frames: int
    signature: str | None
    precision: str
    input_layer: str = ""


@dataclasses.dataclass(frozen=True)
class BenchTarget:
    case: Case
    model: str
    signature: str | None
    note: str = ""


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "runs/litert_vocos_pixel10_gpu_fp16"
OUT_DIR = MODELS_DIR / "android_bench"
INPUTS_DIR = OUT_DIR / "inputs"
LOGS_DIR = OUT_DIR / "logs"
PROFILING_DIR = OUT_DIR / "profiling"
WAVS_DIR = OUT_DIR / "wavs"
PHONE_WAVS_DIR = OUT_DIR / "phone_wavs"
RESULTS_CSV = OUT_DIR / "results.csv"
SUMMARY_MD = OUT_DIR / "summary.md"
PUSHED_FILES_TXT = OUT_DIR / "pushed_files.txt"
DEVICE_INFO_TXT = OUT_DIR / "device_info.txt"

ADB_DEFAULT = "/mnt/c/Users/genei/Downloads/platform-tools/adb.exe"
APK_DEFAULT = ROOT / "tools/android/android_aarch64_benchmark_model.apk"
REMOTE = "/data/local/tmp/kokoro_vocos_bench"
ACTIVITY = "org.tensorflow.lite.benchmark/.BenchmarkModelActivity"

RESULT_FIELDS = [
    "timestamp",
    "device_serial",
    "flavor",
    "model",
    "benchmark_model",
    "sample",
    "frames",
    "signature",
    "benchmark_signature",
    "run_index",
    "status",
    "delegate",
    "precision",
    "input_shape",
    "output_shape",
    "init_ms",
    "first_inference_ms",
    "avg_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
    "raw_log",
    "warmup_iterations",
    "measured_iterations",
    "benchmark_note",
    "error",
]


CASES: list[Case] = [
    Case("fp32_fixed", "vocos_fp32_for_fp16_af_alloy_00001_00_330f_litert.tflite", "af_alloy_00001_00", 330, None, "fp32"),
    Case("fp32_fixed", "vocos_fp32_for_fp16_af_alloy_00002_00_1094f_litert.tflite", "af_alloy_00002_00", 1094, None, "fp32"),
    Case("fp32_fixed", "vocos_fp32_for_fp16_af_alloy_00403_00_1754f_litert.tflite", "af_alloy_00403_00", 1754, None, "fp32"),
    Case("fp16_fixed", "vocos_fp16_af_alloy_00001_00_330f_litert.tflite", "af_alloy_00001_00", 330, None, "fp16"),
    Case("fp16_fixed", "vocos_fp16_af_alloy_00002_00_1094f_litert.tflite", "af_alloy_00002_00", 1094, None, "fp16"),
    Case("fp16_fixed", "vocos_fp16_af_alloy_00403_00_1754f_litert.tflite", "af_alloy_00403_00", 1754, None, "fp16"),
    Case("fp16_dynamic", "vocos_fp16_dynamic_litert.tflite", "af_alloy_00001_00", 330, None, "fp16"),
    Case("fp16_dynamic", "vocos_fp16_dynamic_litert.tflite", "af_alloy_00002_00", 1094, None, "fp16"),
    Case("fp16_dynamic", "vocos_fp16_dynamic_litert.tflite", "af_alloy_00403_00", 1754, None, "fp16"),
    Case("fp16_multisig", "vocos_fp16_multisig_static_litert.tflite", "af_alloy_00001_00", 330, "frames_330", "fp16"),
    Case("fp16_multisig", "vocos_fp16_multisig_static_litert.tflite", "af_alloy_00002_00", 1094, "frames_1094", "fp16"),
    Case("fp16_multisig", "vocos_fp16_multisig_static_litert.tflite", "af_alloy_00403_00", 1754, "frames_1754", "fp16"),
]

WAVEHAX_CASES: list[Case] = [
    Case("wavehax_fp16_fixed", "wavehax_fp16_af_alloy_00001_00_330f_litert.tflite", "af_alloy_00001_00", 330, None, "fp16"),
    Case("wavehax_fp16_fixed", "wavehax_fp16_af_alloy_00002_00_1094f_litert.tflite", "af_alloy_00002_00", 1094, None, "fp16"),
    Case("wavehax_fp16_fixed", "wavehax_fp16_af_alloy_00403_00_1754f_litert.tflite", "af_alloy_00403_00", 1754, None, "fp16"),
    Case("wavehax_fp16_dynamic", "wavehax_fp16_dynamic_litert.tflite", "af_alloy_00001_00", 330, None, "fp16"),
    Case("wavehax_fp16_dynamic", "wavehax_fp16_dynamic_litert.tflite", "af_alloy_00002_00", 1094, None, "fp16"),
    Case("wavehax_fp16_dynamic", "wavehax_fp16_dynamic_litert.tflite", "af_alloy_00403_00", 1754, None, "fp16"),
    Case("wavehax_fp16_multisig", "wavehax_fp16_multisig_static_litert.tflite", "af_alloy_00001_00", 330, "frames_330", "fp16"),
    Case("wavehax_fp16_multisig", "wavehax_fp16_multisig_static_litert.tflite", "af_alloy_00002_00", 1094, "frames_1094", "fp16"),
    Case("wavehax_fp16_multisig", "wavehax_fp16_multisig_static_litert.tflite", "af_alloy_00403_00", 1754, "frames_1754", "fp16"),
]


def configure_family(family: str) -> list[Case]:
    global MODELS_DIR, OUT_DIR, INPUTS_DIR, LOGS_DIR, PROFILING_DIR, WAVS_DIR, PHONE_WAVS_DIR
    global RESULTS_CSV, SUMMARY_MD, PUSHED_FILES_TXT, DEVICE_INFO_TXT, REMOTE
    if family == "vocos":
        cases = CASES
        MODELS_DIR = ROOT / "runs/litert_vocos_pixel10_gpu_fp16"
        REMOTE = "/data/local/tmp/kokoro_vocos_bench"
    elif family == "wavehax":
        cases = WAVEHAX_CASES
        MODELS_DIR = ROOT / "runs/wavehax"
        REMOTE = "/data/local/tmp/kokoro_wavehax_bench"
    else:
        raise ValueError(f"Unsupported benchmark family: {family}")

    OUT_DIR = MODELS_DIR / "android_bench"
    INPUTS_DIR = OUT_DIR / "inputs"
    LOGS_DIR = OUT_DIR / "logs"
    PROFILING_DIR = OUT_DIR / "profiling"
    WAVS_DIR = OUT_DIR / "wavs"
    PHONE_WAVS_DIR = OUT_DIR / "phone_wavs"
    RESULTS_CSV = OUT_DIR / "results.csv"
    SUMMARY_MD = OUT_DIR / "summary.md"
    PUSHED_FILES_TXT = OUT_DIR / "pushed_files.txt"
    DEVICE_INFO_TXT = OUT_DIR / "device_info.txt"
    return cases


def fixed_sample_model_name(case: Case) -> str | None:
    candidates: list[str] = []
    if case.model.startswith("wavehax_"):
        candidates.append(f"wavehax_fp16_{case.sample}_{case.frames}f_litert.tflite")
    elif case.model.startswith("vocos_"):
        candidates.append(f"vocos_fp16_{case.sample}_{case.frames}f_litert.tflite")
    for name in candidates:
        if (MODELS_DIR / name).exists():
            return name
    return None


def resolve_benchmark_targets(cases: list[Case]) -> list[BenchTarget]:
    targets: list[BenchTarget] = []
    for case in cases:
        fallback = fixed_sample_model_name(case)
        if case.signature and fallback:
            targets.append(
                BenchTarget(
                    case=case,
                    model=fallback,
                    signature=None,
                    note=f"benchmark_model_apk_cannot_select_signature; using fixed export for {case.signature}",
                )
            )
        elif not (MODELS_DIR / case.model).exists() and fallback:
            targets.append(
                BenchTarget(
                    case=case,
                    model=fallback,
                    signature=None,
                    note=f"requested model missing ({case.model}); using fixed export",
                )
            )
        else:
            targets.append(BenchTarget(case=case, model=case.model, signature=case.signature))
    return targets


def _compose_features_from_pt(path: Path) -> torch.Tensor:
    row = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(row, Mapping):
        raise TypeError(f"Expected mapping in {path}, got {type(row)}")
    asr = row["asr"].float()
    f0 = row["f0"].float()
    noise = row["noise"].float()
    style = row["style"].float()
    total_frames = int(f0.shape[-1])
    if asr.shape[-1] != total_frames:
        asr = torch.nn.functional.interpolate(
            asr.unsqueeze(0), size=total_frames, mode="linear", align_corners=False
        ).squeeze(0)
    return torch.cat(
        [
            asr[:, :total_frames],
            f0[:total_frames].unsqueeze(0),
            noise[:total_frames].unsqueeze(0),
            style.unsqueeze(-1).expand(style.shape[0], total_frames),
        ],
        dim=0,
    )


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc


def adb_base(adb: str, serial: str | None) -> list[str]:
    cmd = [adb]
    if serial:
        cmd.extend(["-s", serial])
    return cmd


def adb(adb_path: str, serial: str | None, args: list[str], check: bool = True) -> str:
    return run([*adb_base(adb_path, serial), *args], check=check).stdout


def adb_shell(adb_path: str, serial: str | None, shell_cmd: str, check: bool = True) -> str:
    return adb(adb_path, serial, ["shell", shell_cmd], check=check)


def parse_timing_metrics(log_text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    inf_line = re.search(
        r"Inference timings in us:\s*Init:\s*([0-9.eE+\-]+),\s*First inference:\s*([0-9.eE+\-]+),.*Inference \(avg\):\s*([0-9.eE+\-]+)",
        log_text,
    )
    if inf_line:
        metrics["init_ms"] = f"{float(inf_line.group(1)) / 1000.0:.6f}"
        metrics["first_inference_ms"] = f"{float(inf_line.group(2)) / 1000.0:.6f}"
        metrics["avg_ms"] = f"{float(inf_line.group(3)) / 1000.0:.6f}"

    count_lines = re.findall(r"count=\d+[^\n]*", log_text)
    if count_lines:
        line = count_lines[-1]
        for key, field in (
            ("min", "min_ms"),
            ("max", "max_ms"),
            ("median", "p50_ms"),
            ("p95", "p95_ms"),
            ("avg", "avg_ms"),
            ("first", "first_inference_ms"),
        ):
            m = re.search(rf"{key}=([0-9.eE+\-]+)", line)
            if m:
                metrics[field] = f"{float(m.group(1)) / 1000.0:.6f}"
    return metrics


def has_success(log_text: str) -> bool:
    return "Inference timings in us:" in log_text


def detect_error(log_text: str) -> str:
    error_patterns = [
        r"Failed to apply GPU delegate\.",
        r"TfLiteGpuDelegate .* not initialized",
        r"failed to delegate .*",
        r"Failed to allocate tensors!",
        r"Unconsumed cmdline flags: .*signature_to_run",
        r"Node number \d+ .* failed to prepare\.",
    ]
    for pat in error_patterns:
        m = re.search(pat, log_text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return ""


def log_indicates_completion(log_text: str) -> bool:
    if "Inference timings in us:" in log_text:
        return True
    terminal_markers = (
        "Failed to apply GPU delegate.",
        "Failed to allocate tensors!",
        "failed to prepare.",
        "Unconsumed cmdline flags:",
    )
    return any(marker in log_text for marker in terminal_markers)


def wait_for_benchmark_log(
    adb_path: str,
    serial: str | None,
    timeout_seconds: int,
    poll_seconds: float = 1.5,
) -> tuple[str, bool]:
    deadline = time.time() + max(1, int(timeout_seconds))
    last_log = ""
    while time.time() < deadline:
        last_log = adb(
            adb_path,
            serial,
            ["logcat", "-d", "-s", "tflite:I", "tflite_BenchmarkModelActivity:I", "*:S"],
            check=False,
        )
        if log_indicates_completion(last_log):
            return last_log, True
        time.sleep(max(0.2, poll_seconds))
    if not last_log:
        last_log = adb(
            adb_path,
            serial,
            ["logcat", "-d", "-s", "tflite:I", "tflite_BenchmarkModelActivity:I", "*:S"],
            check=False,
        )
    return last_log, False


def discover_input_layer(model_path: Path, signature: str | None = None) -> str:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path))
    signatures = interpreter.get_signature_list()
    if signatures:
        sig_key = signature or next(iter(signatures.keys()))
        runner = interpreter.get_signature_runner(sig_key)
        input_details = runner.get_input_details()
        return next(iter(input_details.keys()))
    return str(interpreter.get_input_details()[0]["name"])


def build_bench_args(
    case: Case,
    remote_model: str,
    remote_input: str,
    warmup: int,
    runs: int,
    input_layer: str,
    signature: str | None = None,
) -> str:
    flags = [
        f"--graph={remote_model}",
        f"--warmup_runs={warmup}",
        f"--num_runs={runs}",
        "--use_gpu=true",
        "--gpu_precision_loss_allowed=true",
        "--use_xnnpack=false",
        "--gpu_backend=cl",
        f"--input_layer={input_layer}",
        f"--input_layer_shape=1,642,{case.frames}",
        f"--input_layer_value_files={input_layer}:{remote_input}",
    ]
    if signature:
        flags.append(f"--signature_to_run={signature}")
    return " ".join(flags)


def start_benchmark_activity(adb_path: str, serial: str | None, bench_args: str) -> str:
    cmd = f"am start -S -n {ACTIVITY} --es args {shlex.quote(bench_args)}"
    return adb_shell(adb_path, serial, cmd, check=False)


def save_wav_16bit(path: Path, audio: np.ndarray, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, pcm16)


def run_tflite_inference(
    model_path: Path,
    input_np: np.ndarray,
    signature: str | None = None,
) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path))
    signatures = interpreter.get_signature_list()
    if signatures:
        sig_key = signature or next(iter(signatures.keys()))
        runner = interpreter.get_signature_runner(sig_key)
        input_details = runner.get_input_details()
        output_details = runner.get_output_details()
        in_name = next(iter(input_details.keys()))
        out_name = next(iter(output_details.keys()))
        out = runner(**{in_name: input_np})[out_name]
        return np.asarray(out, dtype=np.float32)

    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_detail["index"], input_np.astype(np.float32))
    interpreter.invoke()
    return np.asarray(interpreter.get_tensor(output_detail["index"]), dtype=np.float32)


def save_host_preview_wavs(input_bins: dict[str, Path], cases: list[Case]) -> None:
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, str]] = set()
    for case in cases:
        key = (case.model, case.sample, case.signature or "")
        if key in seen:
            continue
        seen.add(key)
        model_path = MODELS_DIR / case.model
        if not model_path.exists():
            continue
        bin_path = input_bins[case.sample]
        x = np.fromfile(bin_path, dtype=np.float32).reshape(1, 642, case.frames)
        wav_name = f"{case.flavor}__{case.sample}__{Path(case.model).stem}.wav"
        out_wav = WAVS_DIR / wav_name
        try:
            y = run_tflite_inference(model_path, x, signature=case.signature)
            save_wav_16bit(out_wav, y.reshape(-1), sample_rate=24000)
        except Exception as exc:
            (WAVS_DIR / f"{wav_name}.error.txt").write_text(str(exc) + "\n", encoding="utf-8")


def pull_phone_audio_files(adb_path: str, serial: str | None) -> list[Path]:
    PHONE_WAVS_DIR.mkdir(parents=True, exist_ok=True)
    find_cmd = f"find {REMOTE} -type f \\( -name '*.wav' -o -name '*.pcm' \\)"
    listing = adb_shell(adb_path, serial, find_cmd, check=False)
    remote_files = [line.strip() for line in listing.splitlines() if line.strip().startswith(f"{REMOTE}/")]
    pulled: list[Path] = []
    pull_errors: list[str] = []
    for remote_file in remote_files:
        base = Path(remote_file).name[:180]
        local_path = PHONE_WAVS_DIR / base
        out = adb(adb_path, serial, ["pull", remote_file, str(local_path)], check=False)
        if "cannot connect to daemon" in out.lower() or "error:" in out.lower():
            pull_errors.append(out.strip())
        if local_path.exists():
            pulled.append(local_path)
    if not pulled:
        msg = [
            f"No phone-side WAV/PCM artifacts were collected from {REMOTE}.",
            "TensorFlow Lite benchmark APK typically does not emit output audio files.",
        ]
        if not remote_files:
            msg.append("`find` returned no WAV/PCM files under the remote bench directory.")
        if pull_errors:
            msg.append("")
            msg.append("Pull errors:")
            msg.extend(pull_errors[:5])
        (PHONE_WAVS_DIR / "README.txt").write_text("\n".join(msg) + "\n", encoding="utf-8")
    return pulled


def cleanup_remote(adb_path: str, serial: str | None) -> None:
    adb_shell(adb_path, serial, "cmd power set-fixed-performance-mode-enabled false", check=False)
    adb_shell(adb_path, serial, "settings put global airplane_mode_on 0", check=False)
    adb_shell(adb_path, serial, f"rm -rf {REMOTE}", check=False)


def main() -> None:
    parser = argparse.ArgumentParser("Run Pixel 10 Android benchmark matrix from android/PLAN_BENCH.md")
    parser.add_argument("--family", choices=("vocos", "wavehax"), default="vocos")
    parser.add_argument("--adb", default=ADB_DEFAULT)
    parser.add_argument("--adb-serial", default=None)
    parser.add_argument("--apk", type=Path, default=APK_DEFAULT)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("--measured-iterations-1754", type=int, default=10)
    parser.add_argument("--run-count", type=int, default=3)
    parser.add_argument("--log-wait-seconds", type=int, default=30)
    parser.add_argument("--log-wait-seconds-1754", type=int, default=70)
    parser.add_argument("--case-timeout-seconds", type=int, default=30)
    parser.add_argument("--case-timeout-seconds-1754", type=int, default=60)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-host-wavs", action="store_true")
    parser.add_argument("--skip-phone-audio-pull", action="store_true")
    parser.add_argument("--keep-remote", action="store_true")
    args = parser.parse_args()
    cases = configure_family(args.family)
    targets = resolve_benchmark_targets(cases)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PROFILING_DIR.mkdir(parents=True, exist_ok=True)
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    PHONE_WAVS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    serial = args.adb_serial
    if not serial:
        serial = adb(args.adb, None, ["get-serialno"]).strip()
        if serial in {"unknown", ""}:
            serial = None

    state = adb(args.adb, serial, ["get-state"]).strip()
    if state != "device":
        raise RuntimeError(f"ADB device state is not 'device': {state}")

    adb_shell(args.adb, serial, f"mkdir -p {REMOTE}/models {REMOTE}/inputs {REMOTE}/logs {REMOTE}/profiles")

    getprop = adb_shell(args.adb, serial, "getprop")
    mem_cpu = adb_shell(args.adb, serial, "cat /proc/meminfo; echo; cat /proc/cpuinfo")
    DEVICE_INFO_TXT.write_text(getprop + "\n\n" + mem_cpu, encoding="utf-8")

    adb_shell(args.adb, serial, "cmd power set-fixed-performance-mode-enabled true", check=False)
    adb_shell(args.adb, serial, "settings put global airplane_mode_on 1", check=False)

    pt_map = {
        "af_alloy_00001_00": (ROOT / "data/af_alloy_00001_00.pt", 330),
        "af_alloy_00002_00": (ROOT / "data/af_alloy_00002_00.pt", 1094),
        "af_alloy_00403_00": (ROOT / "data/af_alloy_00403_00.pt", 1754),
    }
    input_bins: dict[str, Path] = {}
    for sample, (pt_path, frames) in pt_map.items():
        feat = _compose_features_from_pt(pt_path).float().unsqueeze(0).contiguous().numpy().astype(np.float32)
        if feat.shape != (1, 642, frames):
            raise RuntimeError(f"Unexpected shape for {sample}: {feat.shape}")
        out = INPUTS_DIR / f"{sample}_{frames}f_float32.bin"
        feat.tofile(out)
        input_bins[sample] = out

    pushed_lines: list[str] = []
    unique_models = sorted({t.model for t in targets})
    try:
        for target in targets:
            if target.note:
                pushed_lines.append(
                    f"BENCHMARK_TARGET {target.case.flavor} {target.case.sample}: "
                    f"logical={target.case.model} signature={target.case.signature or ''} "
                    f"benchmark={target.model} benchmark_signature={target.signature or ''} note={target.note}"
                )
        available_models: set[str] = set()
        input_layers: dict[tuple[str, str], str] = {}
        missing_models: set[str] = set()
        for model in unique_models:
            local = MODELS_DIR / model
            if not local.exists():
                missing_models.add(model)
                pushed_lines.append(f"MISSING_LOCAL_MODEL {local}")
                continue
            remote = f"{REMOTE}/models/{model}"
            out = adb(args.adb, serial, ["push", str(local), remote]).strip()
            pushed_lines.append(out)
            available_models.add(model)
        for target in targets:
            local = MODELS_DIR / target.model
            if local.exists():
                try:
                    input_layers[(target.model, target.signature or "")] = discover_input_layer(local, signature=target.signature)
                except Exception as exc:
                    input_layers[(target.model, target.signature or "")] = target.case.input_layer or "serving_default_args_0"
                    pushed_lines.append(f"INPUT_LAYER_DISCOVERY_FAILED {local}: {type(exc).__name__}: {exc}")
        for sample, local in input_bins.items():
            remote = f"{REMOTE}/inputs/{local.name}"
            out = adb(args.adb, serial, ["push", str(local), remote]).strip()
            pushed_lines.append(out)
        PUSHED_FILES_TXT.write_text("\n".join(pushed_lines) + "\n", encoding="utf-8")

        pkg_check = adb_shell(args.adb, serial, "cmd package path org.tensorflow.lite.benchmark", check=False)
        if "package:" not in pkg_check:
            run([*adb_base(args.adb, serial), "install", "-r", "-d", "-g", str(args.apk.resolve())], check=True)

        results: list[dict[str, str]] = []
        headers = ",".join(RESULT_FIELDS)
        if RESULTS_CSV.exists():
            existing_header = RESULTS_CSV.read_text(encoding="utf-8").splitlines()[:1]
            if existing_header and existing_header[0] != headers:
                backup = RESULTS_CSV.with_name(
                    f"{RESULTS_CSV.stem}_legacy_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}{RESULTS_CSV.suffix}"
                )
                RESULTS_CSV.rename(backup)
        if not RESULTS_CSV.exists():
            RESULTS_CSV.write_text(headers + "\n", encoding="utf-8")

        smoke_target = next((target for target in targets if target.case.frames == 330 and target.signature is None), targets[0])
        smoke_case = smoke_target.case
        smoke_remote_model = f"{REMOTE}/models/{smoke_target.model}"
        smoke_remote_input = f"{REMOTE}/inputs/{input_bins[smoke_case.sample].name}"
        smoke_input_layer = input_layers.get((smoke_target.model, smoke_target.signature or ""), smoke_case.input_layer or "serving_default_args_0")
        smoke_args = build_bench_args(
            smoke_case,
            smoke_remote_model,
            smoke_remote_input,
            warmup=1,
            runs=1,
            input_layer=smoke_input_layer,
            signature=smoke_target.signature,
        )
        adb(args.adb, serial, ["logcat", "-c"])
        start_benchmark_activity(args.adb, serial, smoke_args)
        smoke_log, _ = wait_for_benchmark_log(
            args.adb,
            serial,
            timeout_seconds=max(10, args.case_timeout_seconds),
        )
        (LOGS_DIR / "smoke_fp16_fixed__af_alloy_00001_00__run_00.txt").write_text(smoke_log, encoding="utf-8")

        run_targets = targets if not args.smoke_only else [smoke_target]

        for target in run_targets:
            case = target.case
            for run_index in range(1, args.run_count + 1):
                local_log = LOGS_DIR / f"{case.flavor}__{case.sample}__run_{run_index:02d}.txt"
                remote_model = f"{REMOTE}/models/{target.model}"
                remote_input = f"{REMOTE}/inputs/{input_bins[case.sample].name}"
                input_layer = input_layers.get((target.model, target.signature or ""), case.input_layer or "serving_default_args_0")
                measured = args.measured_iterations_1754 if case.frames >= 1754 else args.measured_iterations
                wait_seconds = args.log_wait_seconds_1754 if case.frames >= 1754 else args.log_wait_seconds
                timeout_seconds = args.case_timeout_seconds_1754 if case.frames >= 1754 else args.case_timeout_seconds

                row = {
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    "device_serial": serial or "",
                    "flavor": case.flavor,
                    "model": case.model,
                    "benchmark_model": target.model,
                    "sample": case.sample,
                    "frames": str(case.frames),
                    "signature": case.signature or "",
                    "benchmark_signature": target.signature or "",
                    "run_index": str(run_index),
                    "status": "",
                    "delegate": "gpu_cl",
                    "precision": case.precision,
                    "input_shape": f"1,642,{case.frames}",
                    "output_shape": "",
                    "init_ms": "",
                    "first_inference_ms": "",
                    "avg_ms": "",
                    "p50_ms": "",
                    "p90_ms": "",
                    "p95_ms": "",
                    "p99_ms": "",
                    "min_ms": "",
                    "max_ms": "",
                    "raw_log": str(local_log.relative_to(ROOT)),
                    "warmup_iterations": str(args.warmup_iterations),
                    "measured_iterations": str(measured),
                    "benchmark_note": target.note,
                    "error": "",
                }

                if target.model not in available_models:
                    msg = "skipped_missing_model"
                    local_log.write_text(msg + "\n", encoding="utf-8")
                    row["status"] = msg
                    row["error"] = f"Missing local benchmark model file: {MODELS_DIR / target.model}"
                    results.append(row)
                else:
                    bench_args = build_bench_args(
                        case,
                        remote_model,
                        remote_input,
                        warmup=args.warmup_iterations,
                        runs=measured,
                        input_layer=input_layer,
                        signature=target.signature,
                    )
                    adb(args.adb, serial, ["logcat", "-c"], check=False)
                    start_benchmark_activity(args.adb, serial, bench_args)
                    log_text, finished = wait_for_benchmark_log(
                        args.adb,
                        serial,
                        timeout_seconds=max(wait_seconds, timeout_seconds),
                    )
                    local_log.write_text(
                        f"$ adb shell am start -S -n {ACTIVITY} --es args \"{bench_args}\"\n\n{log_text}",
                        encoding="utf-8",
                    )

                    metrics = parse_timing_metrics(log_text)
                    for k, v in metrics.items():
                        if k in row:
                            row[k] = v
                    if has_success(log_text) or row["avg_ms"]:
                        row["status"] = "ok"
                        if not has_success(log_text):
                            row["error"] = "completion_marker_timeout_but_metrics_present"
                    else:
                        row["status"] = "failed"
                        row["error"] = detect_error(log_text)
                        if not finished and not row["error"]:
                            row["error"] = f"timeout_waiting_for_benchmark_log_{max(wait_seconds, timeout_seconds)}s"
                    results.append(row)

                with RESULTS_CSV.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
                    writer.writerow(row)

        if not args.skip_host_wavs:
            save_host_preview_wavs(input_bins, cases)

        if not args.skip_phone_audio_pull:
            pull_phone_audio_files(args.adb, serial)

        success = sum(1 for r in results if r["status"] == "ok")
        failed = sum(1 for r in results if r["status"] == "failed")
        skipped = sum(1 for r in results if r["status"].startswith("skipped"))
        lines = [
            "# Android Benchmark Summary",
            "",
            f"- Timestamp: {timestamp}",
            f"- Device serial: {serial or 'unknown'}",
            f"- Family: {args.family}",
            f"- Models dir: `{MODELS_DIR.relative_to(ROOT)}`",
            f"- Total runs recorded: {len(results)}",
            f"- Success: {success}",
            f"- Failed: {failed}",
            f"- Skipped: {skipped}",
            "",
            "## Notes",
            "",
            "- Benchmarks were run with TensorFlow Lite benchmark APK + GPU delegate flags.",
            "- `benchmark_model` is the actual APK-compatible file pushed to the phone.",
            "- Multi-signature rows use fixed-shape per-sample exports because the Android benchmark APK cannot select TFLite signatures.",
            "- Missing dynamic models use fixed-shape per-sample exports when available.",
            "- Per-run raw logs are under `android_bench/logs/` and CSV rows are in `android_bench/results.csv`.",
            "- Host-generated inspection WAVs are under `android_bench/wavs/`.",
            "- Phone-side audio artifacts (if any) are under `android_bench/phone_wavs/`.",
        ]
        SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        if not args.keep_remote:
            cleanup_remote(args.adb, serial)


if __name__ == "__main__":
    main()
