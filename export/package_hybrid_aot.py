"""
Assemble the accepted hybrid Kokoro TFLite pipeline and AOT-compile NPU-ready
artifacts for Google Tensor G5.

The hybrid pipeline is intentionally a split model package:
  - BERT and decoder reuse the existing litert-torch multi-signature exports.
  - Non-recurrent hybrid blocks are litert-torch multi-signature exports.
  - LSTM blocks are TensorFlow/Keras TFLite recurrent WHILE subgraphs and are
    kept as fallback artifacts for CPU/GPU execution.

Run:
  uv run python export/package_hybrid_aot.py

Outputs:
  outputs/<git_hash>/hybrid_package/
    fp32/*.tflite
    aot/<component>/*_Google_Tensor_G5.tflite
    manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SDK_TAR = Path("litert_npu/litert_plugin_compiler.tar.gz")


@dataclass(frozen=True)
class Component:
    name: str
    source: Path
    role: str
    signatures: list[str]
    aot: bool
    notes: str


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def get_signatures(path: Path) -> list[str]:
    from ai_edge_litert import interpreter

    interp = interpreter.Interpreter(model_path=str(path))
    return sorted(interp.get_signature_list())


def ops_for_single_model(path: Path) -> list[str]:
    from ai_edge_litert import interpreter

    interp = interpreter.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return [op["op_name"] for op in interp._get_ops_details()]


def build_components(hash_: str) -> list[Component]:
    hybrid = Path("outputs") / hash_ / "hybrid_fused_lstm"
    if not hybrid.exists():
        raise FileNotFoundError(
            f"Missing {hybrid}. Run: uv run python export/export_hybrid_fused_lstm_pipeline.py"
        )

    components: list[Component] = [
        Component(
            "bert",
            Path("outputs/e7de808/kokoro_bert_multisig_fp32.tflite"),
            "bert_encoder",
            ["bert_short", "bert_medium", "bert_long", "bert_max"],
            True,
            "Existing litert-torch multi-signature BERT export.",
        ),
        Component(
            "decoder",
            Path("outputs/c46f2e1/kokoro_decoder_multisig_fp32.tflite"),
            "waveform_decoder",
            ["decoder_short", "decoder_medium", "decoder_long"],
            True,
            "Existing litert-torch multi-signature decoder export; compile with minimal sharding.",
        ),
        Component(
            "text_frontend",
            hybrid / "kokoro_hybrid_text_frontend_fp32.tflite",
            "text_encoder_nonrecurrent",
            get_signatures(hybrid / "kokoro_hybrid_text_frontend_fp32.tflite"),
            True,
            "Embedding + CNN frontend before text LSTM.",
        ),
        Component(
            "text_post",
            hybrid / "kokoro_hybrid_text_post_fp32.tflite",
            "text_encoder_nonrecurrent",
            get_signatures(hybrid / "kokoro_hybrid_text_post_fp32.tflite"),
            True,
            "Transpose/mask after text LSTM.",
        ),
        Component(
            "duration_ada_0",
            hybrid / "kokoro_hybrid_duration_ada_0_ct_fp32.tflite",
            "duration_nonrecurrent",
            get_signatures(hybrid / "kokoro_hybrid_duration_ada_0_ct_fp32.tflite"),
            True,
            "DurationEncoder AdaLayerNorm/style concat block 0.",
        ),
        Component(
            "duration_ada_1",
            hybrid / "kokoro_hybrid_duration_ada_1_ct_fp32.tflite",
            "duration_nonrecurrent",
            get_signatures(hybrid / "kokoro_hybrid_duration_ada_1_ct_fp32.tflite"),
            True,
            "DurationEncoder AdaLayerNorm/style concat block 1.",
        ),
        Component(
            "duration_ada_2",
            hybrid / "kokoro_hybrid_duration_ada_2_ct_fp32.tflite",
            "duration_nonrecurrent",
            get_signatures(hybrid / "kokoro_hybrid_duration_ada_2_ct_fp32.tflite"),
            True,
            "DurationEncoder AdaLayerNorm/style concat block 2.",
        ),
        Component(
            "duration_proj",
            hybrid / "kokoro_hybrid_duration_proj_fp32.tflite",
            "duration_nonrecurrent",
            get_signatures(hybrid / "kokoro_hybrid_duration_proj_fp32.tflite"),
            True,
            "Duration projection logits.",
        ),
        Component(
            "f0n_heads",
            hybrid / "kokoro_hybrid_f0n_heads_fp32.tflite",
            "f0n_nonrecurrent",
            get_signatures(hybrid / "kokoro_hybrid_f0n_heads_fp32.tflite"),
            True,
            "F0/N convolutional heads after shared LSTM.",
        ),
    ]

    for path in sorted(hybrid.glob("kokoro_hybrid_*lstm*_T*_fp32.tflite")):
        ops = ops_for_single_model(path)
        components.append(
            Component(
                path.stem.removeprefix("kokoro_hybrid_").removesuffix("_fp32"),
                path,
                "recurrent_fallback",
                [],
                False,
                f"TensorFlow/Keras recurrent model; ops={','.join(ops)}. Kept out of Tensor G5 AOT.",
            )
        )
    return components


def copy_fp32_components(components: list[Component], fp32_dir: Path) -> dict[str, str]:
    fp32_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for component in components:
        if not component.source.exists():
            raise FileNotFoundError(component.source)
        dest = fp32_dir / f"{component.name}.tflite"
        shutil.copy2(component.source, dest)
        copied[component.name] = str(dest)
    return copied


def aot_compile_component(component: Component, fp32_path: Path, aot_root: Path) -> dict[str, object]:
    from ai_edge_litert.aot import aot_compile as aot_lib
    from ai_edge_litert.aot.vendors.google_tensor import target as gt_target

    out_dir = aot_root / component.name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "compilation_report.txt"
    status_path = out_dir / "status.json"

    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if existing.get("status") == "success" and existing.get("outputs"):
                print(f"AOT reuse: {component.name}")
                return existing
        except Exception:
            pass

    sharding = "minimal" if component.name == "decoder" else "extensive"
    tensor_g5 = gt_target.Target(gt_target.SocModel.TENSOR_G5)
    started = time.time()
    print(f"AOT compile: {component.name} ({fp32_path}) sharding={sharding}")
    result: dict[str, object] = {
        "component": component.name,
        "input": str(fp32_path),
        "out_dir": str(out_dir),
        "sharding": sharding,
        "status": "unknown",
    }
    try:
        compiled = aot_lib.aot_compile(
            str(fp32_path),
            target=[tensor_g5],
            keep_going=True,
            google_tensor_truncation_type="half",
            google_tensor_int64_to_int32=True,
            google_tensor_sharding_intensity=sharding,
        )
        report = compiled.compilation_report()
        report_path.write_text(str(report), encoding="utf-8")
        compiled.export(str(out_dir), model_name=component.name)
        outputs = sorted(str(path) for path in out_dir.glob("*.tflite"))
        status = "success" if outputs else "failed"
        result.update(
            {
                "status": status,
                "elapsed_sec": round(time.time() - started, 3),
                "report": str(report_path),
                "outputs": outputs,
            }
        )
        if outputs:
            print(f"AOT success: {component.name}")
        else:
            result["error"] = "AOT compiler returned without exported .tflite outputs"
            print(f"AOT failed: {component.name}: {result['error']}")
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "elapsed_sec": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"AOT failed: {component.name}: {result['error']}")
    status_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def existing_aot_status(component: Component, aot_root: Path) -> dict[str, object] | None:
    status_path = aot_root / component.name / "status.json"
    if not status_path.exists():
        return None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    outputs = sorted(str(path) for path in (aot_root / component.name).glob("*.tflite"))
    status["outputs"] = outputs
    if status.get("status") == "success" and not outputs:
        status["status"] = "failed"
        status["error"] = status.get("error") or "AOT status had no exported .tflite outputs"
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", default=git_hash())
    parser.add_argument("--skip-aot", action="store_true")
    parser.add_argument(
        "--component",
        action="append",
        help="AOT-compile only this component name. May be passed multiple times.",
    )
    parser.add_argument(
        "--include-decoder",
        action="store_true",
        help="Also compile decoder. Decoder AOT has historically been slow/flaky.",
    )
    args = parser.parse_args()

    hash_ = args.hash
    package_dir = Path("outputs") / hash_ / "hybrid_package"
    fp32_dir = package_dir / "fp32"
    aot_dir = package_dir / "aot"
    package_dir.mkdir(parents=True, exist_ok=True)

    if SDK_TAR.exists():
        os.environ["GOOGLE_TENSOR_SDK_BETA"] = str(SDK_TAR.resolve())

    components = build_components(hash_)
    copied = copy_fp32_components(components, fp32_dir)

    selected = set(args.component or [])
    aot_results = []
    if not args.skip_aot:
        if not SDK_TAR.exists():
            raise FileNotFoundError(SDK_TAR)
        for component in components:
            if not component.aot:
                aot_results.append(
                    {
                        "component": component.name,
                        "status": "skipped",
                        "reason": "recurrent WHILE fallback model",
                    }
                )
                continue
            if component.name == "decoder" and not args.include_decoder:
                aot_results.append(
                    existing_aot_status(component, aot_dir)
                    or {
                        "component": component.name,
                        "status": "skipped",
                        "reason": "decoder AOT omitted unless --include-decoder is set",
                    }
                )
                continue
            if selected and component.name not in selected:
                aot_results.append(
                    existing_aot_status(component, aot_dir)
                    or {
                        "component": component.name,
                        "status": "not_selected",
                    }
                )
                continue
            aot_results.append(
                aot_compile_component(component, Path(copied[component.name]), aot_dir)
            )

    manifest = {
        "git_hash": hash_,
        "package_dir": str(package_dir),
        "fp32_dir": str(fp32_dir),
        "aot_dir": str(aot_dir),
        "sdk_tar": str(SDK_TAR),
        "notes": [
            "Hybrid split-model package accepted after WAV review.",
            "LSTM artifacts are compact recurrent WHILE TFLite models and are not AOT-compiled for Tensor G5.",
            "AOT outputs are per-component Tensor G5 compiled models plus fallback files from the SDK exporter.",
        ],
        "components": [
            {
                **asdict(component),
                "source": str(component.source),
                "packaged_fp32": copied[component.name],
            }
            for component in components
        ],
        "aot_results": aot_results,
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
