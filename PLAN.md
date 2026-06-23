# Plan: fp16 Vocos LiteRT export for Pixel 10 GPU

## Goal

Export the Vocos decoder from `models/vocos/last.pt` as a Pixel 10 friendly fp16 LiteRT model, using real Kokoro feature inputs such as `data/af_alloy_00001_00.pt`, `data/af_alloy_00002_00.pt`, and `data/af_alloy_00403_00.pt`.

The final export should:

- Produce a real fp16 TFLite artifact, not only an fp32 graph with an fp16 filename.
- Avoid or rewrite operations that block LiteRT-Torch conversion, Android GPU delegation, or Google Tensor AOT compilation.
- Preserve audio quality against the current fp32/fp16-weight baseline.
- Save diagnostics for graph ops, tensor dtypes, unsupported ops, Android delegate results, and AOT compiler failures.

## Current state

Working baseline:

- `vocos_export.py --pixel10-fp16-aot` exports fixed-frame models per real input length.
- The final `vocos_fp16_*.tflite` files are now fp16-weight TFLite models produced by:
  1. float32 LiteRT staging export,
  2. AI Edge Quantizer `FLOAT_CASTING` 16-bit weight-only quantization,
  3. final fp16-weight diagnostics.
- Example result for `af_alloy_00001_00`:
  - staging model: about 45 MB,
  - final fp16-weight model: about 26 MB,
  - tensor inventory: `FLOAT16: 38`, `FLOAT32: 447`, `DEQUANTIZE: 38`,
  - local LiteRT inference passes with finite output and RMS around `0.0498`.

Remaining blockers:

- Direct all-fp16 `litert_torch` export still fails in fp16 `Conv1d` lowering:
  - `failed to legalize operation 'tfl.transpose' that was explicitly marked illegal`
  - source: `torch.nn.modules.conv.Conv1d`
- Dynamic-frame export now fails in the backbone linear path:
  - `TracerBoolConversionError`
  - failing graph node: `aten.expand.default(... [1, %sym_size_int_35, 384])`
  - source: `ConvNeXtBlock.pwconv1`
- Google Tensor G5 AOT still fails internally after selecting the full fp16-weight graph:
  - selected `366 / 366 ops`
  - compiler failure type: `INTERNAL`
  - no per-op unsupported list emitted by the compiler.
- Older Android OpenCL GPU delegate logs also showed trouble around:
  - broadcasted `ADD`,
  - `BROADCAST_TO`,
  - broadcasted `MUL`,
  - `SLICE` on 1D/2D tensors.

## Strategy

Do not mutate the training model first. Build an export-only model wrapper in `vocos_export.py` so the checkpoint remains compatible with training code and existing inference code. Once the export-only graph is stable, decide whether any of those rewrites should move into shared model code.

Use three export targets in order:

1. **fp16-weight TFLite baseline**: float32 LiteRT export plus fp16 weight casting. This is already working and should remain the fallback.
2. **GPU-delegate friendly fp16-weight graph**: same precision path, but with graph rewrites that remove unsupported shape/broadcast/slice/transpose patterns.
3. **direct all-fp16 graph**: only pursue after the graph is already GPU-friendly, because LiteRT-Torch currently cannot lower fp16 `Conv1d` cleanly.

## Phase 1: lock the baseline

1. Keep the current working command as the reproducible baseline:

   ```bash
   uv run python vocos_export.py \
     --checkpoint models/vocos/last.pt \
     --output-dir runs/litert_vocos_pixel10_aot \
     --sample-count 3 \
     --pixel10-fp16-aot \
     --dynamic-frames \
     --google-tensor-compiler-lib tools/google_tensor_ml_sdk
   ```

2. Keep generating these diagnostics for every run:

   - `diagnostics/summary.txt`
   - `diagnostics/*_op_inventory.txt`
   - `diagnostics/true_fp16_export_failure.txt`
   - `diagnostics/dynamic_export_failure.txt`
   - `diagnostics/*_aot_google_tensor_g5.txt`
   - copied compiler `*_tmp*.error` logs

3. Add a quality guard:

   - Run local LiteRT inference on each exported fp16 model.
   - Save RMS, peak, finite/NaN status, and output shape.
   - Compare against fp32 staging output with max error, mean absolute error, and RMS delta.

4. Acceptance for Phase 1:

   - All three real examples export.
   - All three fp16-weight models load in LiteRT.
   - Diagnostics confirm `FLOAT16` tensors exist.
   - Local inference is finite and non-silent.

## Phase 2: remove LayerNorm as a built-in op

Already implemented, but keep it as a formal export rewrite:

1. Replace each `nn.LayerNorm(dim)` with `ExportSafeLayerNorm`.
2. Compute normalization with primitive tensor ops:

   - `mean`
   - `sub`
   - `square`
   - `mean`
   - `rsqrt`
   - `mul`
   - optional affine `mul/add`

3. Keep constants in the input dtype:

   - Convert `eps` to `x.dtype` in forward.
   - Store affine weights and bias as module parameters so `.half()` updates them.

4. Verify:

   - No `LayerNorm` or fused layer norm remains in op inventory.
   - Outputs match the original module within fp32 tolerance before quantization.

## Phase 3: remove feature-channel slices

Already implemented, but keep it as a required export rewrite:

1. Replace `KokoroFeatureConditioner.forward()` slicing with full-channel projections.
2. For each branch, widen the first projection to consume the full 642-channel input:

   - ASR projection keeps weights in channels `[0:512]`.
   - F0 projection keeps weights in channel `[512]`.
   - Noise projection keeps weights in channel `[513]`.
   - Style projection keeps weights in channels `[514:642]`.
   - All unrelated input channels are zero-weighted.

3. This avoids `SLICE` ops in the conditioner while preserving exact math.
4. Verify:

   - Conditioner output before/after rewrite matches closely in fp32.
   - The early `SLICE` ops disappear from the op inventory.

## Phase 4: remove unsupported broadcast patterns

Target the broadcast ops seen in Android GPU logs and TFLite inventory:

1. Replace shape-dependent broadcasts in `ExportSafeISTFTHead`:

   - Avoid `BROADCAST_TO` for window/envelope expansion.
   - Pre-shape constants to match the exported fixed frame length when running fixed-frame export.
   - For variable-length export, use operations that do not require Python or JAX boolean conversion on symbolic sizes.

2. Replace affine broadcasts in normalization and ConvNeXt:

   - Express gamma, bias, scale, and shift with explicit reshape to stable ranks.
   - Prefer fixed-rank `[1, C, 1]` or `[1, 1, C]` constants over implicit rank expansion.

3. Check every `ADD` and `MUL` with mismatched ranks:

   - Identify tensor ranks from `*_op_inventory.txt`.
   - Rewrite the source module so operands have identical rank before LiteRT conversion.

4. Acceptance:

   - No `BROADCAST_TO` remains in the fixed-frame fp16 model.
   - Android GPU delegate logs no longer report broadcasted `ADD` or `MUL` fallback.

## Phase 5: reduce or remove transpose-heavy Conv1d lowering

The direct all-fp16 path currently fails because fp16 `Conv1d` lowers through a `tfl.transpose` that LiteRT-Torch marks illegal. Fix this in export-only modules before attempting all-fp16 again.

Try these options in order:

1. **Pointwise Linear rewrite**

   - Replace ConvNeXt `pwconv1` and `pwconv2` `nn.Linear` with equivalent 1x1 temporal convolutions in channel-first layout.
   - This removes `[B, T, C] -> Linear -> [B, T, C]` rank expansion from the dynamic export path.
   - Verify whether LiteRT-Torch lowers this more cleanly than dynamic `aten.expand`.

2. **Channel-first ConvNeXt block**

   - Keep the whole backbone in `[B, C, T]`.
   - Rewrite LayerNorm as channel-wise normalization over dimension `C` without transposing to `[B, T, C]`.
   - Replace pointwise linear layers with `Conv1d(kernel_size=1)`.
   - This should remove most ConvNeXt transposes.

3. **Conv2D temporal block**

   - If `Conv1d` still lowers through illegal fp16 transposes, replace temporal convs with export-only `Conv2d` wrappers over a singleton spatial dimension.
   - Inspect the resulting TFLite graph to see whether LiteRT-Torch emits stable `CONV_2D` / `DEPTHWISE_CONV_2D` without illegal transposes.

4. **Weight-only fp16 fallback**

   - If direct all-fp16 still fails after graph cleanup, keep the fp16-weight quantized model as the shipping model.
   - Continue optimizing operation compatibility rather than insisting on all intermediate tensors being `FLOAT16`.

5. Acceptance:

   - Fixed-frame fp16-weight model has fewer `TRANSPOSE` ops than the current baseline.
   - Direct all-fp16 probe either succeeds or has a later, smaller, well-documented blocker.
   - Dynamic export failure no longer points at `ConvNeXtBlock.pwconv1`.

## Phase 6: remove remaining Slice ops where practical

Some `SLICE` ops come from ISTFT/head logic rather than the conditioner.

1. Classify every remaining `SLICE` by source:

   - head `chunk`,
   - real/imag split,
   - DC/Nyquist/mid-bin handling,
   - output trimming,
   - conditioner/backbone leftovers.

2. Replace avoidable slices:

   - Use separate output projections for magnitude and phase instead of `x.chunk(2, dim=1)`.
   - Pre-split static Fourier bases into fixed buffers.
   - Use fixed-frame output trimming constants for fixed-frame exports.

3. Keep only slices that are known to be accepted by Pixel GPU/AOT.

4. Acceptance:

   - Remaining `SLICE` ops are documented with source and tensor rank.
   - Android GPU delegate logs no longer report unsupported 1D/2D `SLICE`.

## Phase 7: variable-length input support

Treat variable length as a separate deliverable after fixed-frame GPU compatibility improves.

1. Keep fixed-frame per-length export as the production fallback:

   - 330 frames
   - 1094 frames
   - 1754 frames
   - any additional bucket lengths needed by streaming.

2. For dynamic export, remove the current symbolic-shape blocker:

   - Replace `nn.Linear` paths that trigger `aten.expand(... %sym_size_int ...)`.
   - Avoid JAX-traced Python boolean conversions.
   - Avoid shape-dependent constant expansion in the model forward.

3. Consider bucketed fixed-shape models before true dynamic TFLite:

   - Buckets are often easier for GPU/AOT compilers.
   - Streaming can route chunks to the nearest bucket or use fixed chunk sizes.

4. Acceptance:

   - Either dynamic TFLite export succeeds and runs locally, or bucketed fixed-shape export is documented as the selected Pixel path.

## Phase 8: Android GPU test loop

For each graph rewrite:

1. Export the fp16-weight model.
2. Push the model and real input binary to the Pixel 10.
3. Run the TensorFlow Lite benchmark APK with OpenCL GPU delegate:

   - `--use_gpu=true`
   - `--gpu_backend=cl`
   - `--gpu_precision_loss_allowed=true`
   - real `--input_layer_value_files`

4. Save logs under `output-dir/android_gpu`.
5. Record:

   - whether delegate creation succeeds,
   - number of ops on GPU vs CPU,
   - unsupported ops,
   - compile time,
   - inference average and p50/p90/p99 if available.

6. Acceptance:

   - GPU delegate applies successfully.
   - No unsupported-op report for known rewritten ops.
   - Audio output is finite and non-silent.

## Phase 9: AOT compile loop

For each graph rewrite:

1. Run Google Tensor G5 AOT compilation with `tools/google_tensor_ml_sdk`.
2. Copy every `/tmp/*.error` log into diagnostics.
3. Record:

   - selected ops / total ops,
   - partition count,
   - failure message,
   - compiler internal error, if any.

4. If AOT still fails internally after selecting all ops:

   - Treat that as a backend compiler issue, not an unsupported-op issue.
   - Keep the Android GPU delegate path as the practical runtime target.
   - File the copied error log upstream if needed.

5. Acceptance:

   - AOT emits a compiled model, or the internal compiler failure is reproducibly documented with the smallest graph that triggers it.

## Validation checklist

Run before each checkpoint:

```bash
uv run python -m py_compile prepare_weights.py vocos_export.py
git diff --check -- prepare_weights.py vocos_export.py PROGRESS.md PLAN.md
```

Run after export:

```bash
uv run python vocos_export.py \
  --checkpoint models/vocos/last.pt \
  --output-dir runs/litert_vocos_pixel10_aot \
  --sample-count 3 \
  --pixel10-fp16-aot \
  --dynamic-frames \
  --google-tensor-compiler-lib tools/google_tensor_ml_sdk
```

Then inspect:

- `runs/litert_vocos_pixel10_aot/diagnostics/summary.txt`
- `runs/litert_vocos_pixel10_aot/diagnostics/*_op_inventory.txt`
- `runs/litert_vocos_pixel10_aot/diagnostics/*_aot_google_tensor_g5*.txt`
- Android GPU benchmark logs when a phone run is requested.

## Definition of done

The fp16 export work is complete when:

- `vocos_fp16_*.tflite` contains actual `FLOAT16` tensors.
- Local LiteRT inference passes on all real examples.
- The model has no known unsupported LayerNorm, conditioner slice, broadcast, or avoidable transpose patterns.
- Pixel 10 GPU delegate applies successfully, or remaining unsupported ops are explicitly identified with source modules and a follow-up patch plan.
- AOT compile either succeeds or has a minimal, reproducible internal compiler failure with logs saved in diagnostics.
