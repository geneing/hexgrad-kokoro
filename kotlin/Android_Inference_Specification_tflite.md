# Kokoro TTS — Android LiteRT Inference Specification

> **Status**: Models validated. Full-pipeline correlation 1.0000 (TFLite vs PyTorch reference).
> **Sample rate**: 24 000 Hz mono float32.
> **Supported text**: English (American, phoneme-based).

---

## 1. Overview

Kokoro TTS inference on Android runs across **7 stages**, 6 of which use LiteRT (TFLite) models.
Stage 3 (duration expansion) and the IRFFT + overlap-add in Stage 7 run in pure Kotlin/Java.

```
Text
 │
 ▼
[S1] BERT Encoder            bert_float32.tflite          25.9 MB
 │  d_en [1,510,512]
 ▼
[S2] Duration Predictor      duration_predictor_float32   54.1 MB
 │  pred_dur[510], d_enc[1,640,510], t_en_static[1,512,510]
 ▼
[S3] Duration Expansion      (Kotlin — no model)
 │  expanded_indices[], T_acoustic
 ▼ index_select on d_enc / t_en_static
[S4] Acoustic Expand         acoustic_expand_float32      7.4 MB
 │  en [1,543,512]
 ▼
[S5] F0/N Predictor          f0n_predictor_float32        27.0 MB
 │  F0_pred[1,1086], N_pred[1,1086]
 ▼
[S6] Vocos Conditioner       vocoder_conditioner_float32  1.1 MB
 │  conditioned[1,192,1086]
 ▼
[S7] Vocos Stream Chunk      vocoder_stream_chunk_float32 32.5 MB
     → IRFFT (Kotlin) + Overlap-Add (Kotlin)
     → audio [float32, 24 kHz]
```

**Total model size**: ~148 MB (float32). Float16 variants also available (see §8).

---

## 2. Constants

```kotlin
const val SAMPLE_RATE        = 24_000
const val MAX_INPUT_LENGTH   = 510        // max padded phoneme token count
const val T_ACOUSTIC         = 543        // acoustic_expand / f0n static time dim
const val T_F0               = 1086       // conditioner time dim (2 × T_ACOUSTIC)
const val VOCOS_CHUNK_FRAMES = 16         // frames per backbone chunk
const val VOCOS_HOP          = 300        // hop length (samples per frame)
const val VOCOS_N_FFT        = 1200       // IRFFT window size
const val VOCOS_TAIL         = 900        // overlap-add tail = N_FFT - HOP
const val VOCOS_EMBED_DIM    = 192        // conditioner output channels
const val VOCOS_STATE_K      = 6          // streaming state time depth
const val VOCOS_IRNC_BINS    = 601        // IRFFT output bins (N_FFT/2 + 1)
```

---

## 3. Voice Pack & Style Vector

Each voice is a precomputed style matrix `[num_phonemes, 256]` stored as a `.pt` file.
At inference, select the row matching `phonemes.size - 1` (0-indexed).

```kotlin
// style: FloatArray of length 256
// style[0..127]  → decoder style (s_ref)
// style[128..255] → predictor style (s_predictor)
val style = loadVoicePack(voiceFile)[phonemes.size - 1]
```

---

## 4. Text → Phoneme → Token IDs

Use the **Misaki g2p** library (Python) or a pre-built phoneme lookup table for English.
The resulting token sequence is prepended/appended with token `0` (BOS/EOS):

```kotlin
val rawIds: IntArray = g2p(text)                   // phoneme → token IDs
val seqLen = rawIds.size + 2                       // +2 for BOS/EOS
val inputIds = IntArray(MAX_INPUT_LENGTH)           // zero-padded
val textMask = FloatArray(MAX_INPUT_LENGTH)         // 1.0 at valid positions
inputIds[0] = 0                                    // BOS
rawIds.copyInto(inputIds, 1)
inputIds[seqLen - 1] = 0                           // EOS
for (i in 0 until seqLen) textMask[i] = 1.0f
```

---

## 5. Stage-by-Stage Inference

### S1 — BERT Encoder

**Model**: `bert_float32.tflite`

| Tensor | Shape | Type |
|--------|-------|------|
| `input_ids` (IN) | [1, 510] | int32 |
| `text_mask` (IN) | [1, 510] | float32 |
| `d_en` (OUT) | [1, 510, 512] | float32 — NTC layout |

```kotlin
bertInterp.setInputTensor("input_ids", inputIds)
bertInterp.setInputTensor("text_mask", textMask)
bertInterp.run()
val dEn: FloatArray = bertInterp.getOutputTensor("d_en")  // [1,510,512] NTC
```

---

### S2 — Duration Predictor

**Model**: `duration_predictor_float32.tflite`
(Re-exported with weight-norm folded: inputs use `serving_default_NAME:0` prefixes, outputs use `StatefulPartitionedCall:N`)

| Tensor | Shape | Type | Note |
|--------|-------|------|------|
| `serving_default_d_en:0` (IN) | [1, 510, 512] | float32 | NTC — from S1 |
| `serving_default_style:0` (IN) | [1, 256] | float32 | voice style |
| `serving_default_text_mask:0` (IN) | [1, 510] | float32 | padding mask |
| `serving_default_speed:0` (IN) | [1] | **float32** | speed factor (default 1.0) |
| `serving_default_input_ids:0` (IN) | [1, 510] | **int64** | phoneme token IDs |
| `StatefulPartitionedCall:0` (OUT) | [1, 640, 510] | float32 | d_enc NCT |
| `StatefulPartitionedCall:1` (OUT) | [510] | float32 | pred_dur |
| `StatefulPartitionedCall:2` (OUT) | [1, 512, 510] | float32 | t_en_static NCT |

> ⚠ **Speed input is float32** (not int32 as in some older exports).  
> ⚠ **input_ids input is int64** (not int32).

Output ordering from `StatefulPartitionedCall:N`:
- `:0` → `d_enc [1, 640, 510]` (predictor text-encoder features, NCT)
- `:1` → `pred_dur [510]` (per-phoneme frame counts)
- `:2` → `t_en_static [1, 512, 510]` (text-encoder features, NCT)

```kotlin
durInterp.setInputTensor("serving_default_d_en:0", dEn)
durInterp.setInputTensor("serving_default_style:0", style)
durInterp.setInputTensor("serving_default_text_mask:0", textMask)
durInterp.setInputTensor("serving_default_speed:0", floatArrayOf(speed.toFloat()))
durInterp.setInputTensor("serving_default_input_ids:0", inputIdsLong)   // Int64!
durInterp.run()
val dEnc:       FloatArray = durInterp.getOutputByIndex(0)  // [1,640,510] NCT
val predDur:    FloatArray = durInterp.getOutputByIndex(2)  // [510]
val tEnStatic:  FloatArray = durInterp.getOutputByIndex(1)  // [1,512,510] NCT
```

---

### S3 — Duration Expansion (Kotlin, no model)

Expand phoneme features to acoustic frame rate using `pred_dur`:

```kotlin
/**
 * Computes expanded_indices from pred_dur (durations for MAX_INPUT_LENGTH phonemes).
 * Returns (expandedIndices, T_acoustic).
 */
fun expandDurations(predDur: FloatArray): Pair<IntArray, Int> {
    val boundaries = FloatArray(predDur.size)
    var cumSum = 0f
    for (i in predDur.indices) {
        cumSum += predDur[i].coerceAtLeast(0f)
        boundaries[i] = cumSum
    }
    val tAcoustic = cumSum.toInt().coerceAtMost(T_ACOUSTIC)
    val expanded = IntArray(tAcoustic)
    for (t in 0 until tAcoustic) {
        var idx = 0
        while (idx < boundaries.size - 1 && boundaries[idx] <= t.toFloat()) idx++
        expanded[t] = idx
    }
    return Pair(expanded, tAcoustic)
}
```

Then index-select on the NCT dimension-2 (T axis):

```kotlin
// dEnc: [1, 640, 510] NCT → dEncExp: [1, 640, T_acoustic] → zero-padded to [1, 640, T_ACOUSTIC]
val dEncExp = indexSelectAndPad(dEnc, expandedIndices, tAcoustic, nChannels=640, tStatic=T_ACOUSTIC)
// tEnStatic: [1, 512, 510] NCT → tEnExp: [1, 512, T_acoustic]  (no padding needed here)
val asr = indexSelectNCT(tEnStatic, expandedIndices, nChannels=512)

// Helper: index-select on axis 2 of [1, C, T_in] → pad to [1, C, T_ACOUSTIC]
fun indexSelectAndPad(nct: FloatArray, idx: IntArray, tAc: Int, nChannels: Int, tStatic: Int): FloatArray {
    val out = FloatArray(nChannels * tStatic)
    for (c in 0 until nChannels) {
        for (t in 0 until tAc) {
            out[c * tStatic + t] = nct[c * 510 + idx[t]]
        }
        // positions tAc..tStatic-1 remain 0 (zero-padded)
    }
    return out  // [C * T_static] = NCT flat
}
```

> **Important**: Pass the **full T_ACOUSTIC d_enc padded tensor** to S4 (not trimmed to tAcoustic).
> The BiLSTM in acoustic_expand needs consistent padding for its backward pass.

---

### S4 — Acoustic Expand

**Model**: `acoustic_expand_float32.tflite`

| Tensor | Shape | Type | Note |
|--------|-------|------|------|
| `d_enc_expanded` (IN) | [1, 543, 640] | float32 | **NTC** — padded dEnc transposed |
| `en` (OUT) | [1, 543, 512] | float32 | **NTC** |

> Note: Input is **NTC** (time first), unlike `d_enc` which is NCT.

```kotlin
// Transpose dEncExp from NCT [1,640,T_ACOUSTIC] to NTC [1,T_ACOUSTIC,640]
val dEncNtc = transposeNCTtoNTC(dEncExp, 640, T_ACOUSTIC)
acexpInterp.setInputTensor("d_enc_expanded", dEncNtc)
acexpInterp.run()
val enNtc: FloatArray = acexpInterp.getOutputTensor("en")  // [1,T_ACOUSTIC,512] NTC

// Transpose to NCT for f0n — do NOT trim to tAcoustic before sending to f0n
val enNct = transposeNTCtoNCT(enNtc, T_ACOUSTIC, 512)  // [1, 512, T_ACOUSTIC] NCT
```

---

### S5 — F0/N Predictor

**Model**: `f0n_predictor_float32.tflite`

| Tensor | Shape | Type | Note |
|--------|-------|------|------|
| `en` (IN) | [1, 512, 543] | float32 | **NCT** — full T_ACOUSTIC (not trimmed) |
| `style` (IN) | [1, 256] | float32 | voice style |
| `F0_pred` (OUT) | [1, 1086] | float32 | fundamental frequency |
| `N_pred` (OUT) | [1, 1086] | float32 | noise level |

```kotlin
f0nInterp.setInputTensor("en", enNct)
f0nInterp.setInputTensor("style", style)
f0nInterp.run()
val f0Full: FloatArray = f0nInterp.getOutputTensor("F0_pred")  // [1,1086]
val nFull:  FloatArray = f0nInterp.getOutputTensor("N_pred")   // [1,1086]

// Trim to actual T_f0 = 2 * tAcoustic
val tF0Actual = 2 * tAcoustic
val f0Pred = f0Full.copyOfRange(0, tF0Actual)   // length tF0Actual
val nPred  = nFull.copyOfRange(0, tF0Actual)    // length tF0Actual
```

---

### S6 — Vocos Conditioner

**Model**: `vocoder_conditioner_float32.tflite`

The conditioner input is a feature concatenation [asr, F0, N, style] of shape [1, 1086, 642]:

| Component | Dims | Source |
|-----------|------|--------|
| ASR (t_en_static expanded) | 512 | `asr[t]` from S3 |
| F0 | 1 | `f0Pred[t]` from S5 |
| N | 1 | `nPred[t]` from S5 |
| Style d_ref (first 128) | 128 | `style[0..127]` |

```kotlin
// Assemble features: [1, 642, T_f0_actual] NCT, then transpose to NTC and pad
val features = FloatArray(T_F0 * 642)  // [T_F0, 642] NTC flat

// asr: [1, 512, T_acoustic] NCT → need asr[c][t] for each frame
for (t in 0 until tF0Actual) {
    val srcT = t / 2  // asr is at acoustic rate, F0/N are at 2x acoustic rate
    // Actually: asr is at T_acoustic rate, F0 at T_f0 = 2*T_acoustic rate
    // The _build_vocos_features function uses F.interpolate to match F0 rate
    // Simplification: index ASR with interpolation or nearest neighbor
}
```

**Feature assembly (exact logic from `_build_vocos_features`):**

```kotlin
fun buildVocosFeatures(
    asr: FloatArray,        // [512, tAcoustic] — asr[c][t] at NCT flat
    f0Pred: FloatArray,     // [tF0Actual]
    nPred: FloatArray,      // [tF0Actual]
    style: FloatArray,      // [256]
    tAcoustic: Int,
    tF0Actual: Int,
    tF0Static: Int = T_F0   // pad to this
): FloatArray {
    val out = FloatArray(tF0Static * 642)
    for (t in 0 until tF0Actual) {
        val offset = t * 642
        // ASR: interpolate from tAcoustic to tF0Actual
        val srcFrac = t * tAcoustic.toFloat() / tF0Actual
        val srcIdx = srcFrac.toInt().coerceIn(0, tAcoustic - 1)
        for (c in 0 until 512) {
            out[offset + c] = asr[c * tAcoustic + srcIdx]
        }
        // F0 (channel 512)
        out[offset + 512] = f0Pred[t]
        // N  (channel 513)
        out[offset + 513] = nPred[t]
        // Style d_ref = style[0..127] (channels 514..641)
        for (c in 0 until 128) {
            out[offset + 514 + c] = style[c]
        }
    }
    // positions tF0Actual..tF0Static-1 remain 0
    return out
}
```

> Note: onnx2tf converted the conditioner input to **NTC [1, 1086, 642]** (onnx2tf `-osd` transposes).

| Tensor | Shape | Type |
|--------|-------|------|
| `features` (IN) | [1, 1086, 642] | float32 — **NTC** |
| `conditioned` (OUT) | [1, 192, 1086] | float32 — **NCT** |

```kotlin
condInterp.setInputTensor("features", featuresNtc)
condInterp.run()
val conditioned: FloatArray = condInterp.getOutputTensor("conditioned")
// conditioned[c][t] = conditioned[c * T_F0 + t], shape [192, T_F0] NCT flat
```

---

### S7 — Vocos Streaming Chunk

**Model**: `vocoder_stream_chunk_float32.tflite`

This model processes `VOCOS_CHUNK_FRAMES = 16` conditioned frames at a time.
It is **stateful** — streaming state must be passed between chunks.

#### Inputs

| Tensor | Shape | Type | Note |
|--------|-------|------|------|
| `conditioned_chunk` (IN) | [1, 16, 192] | float32 | **NTC** — 16 conditioned frames |
| `embed_prev` (IN) | [1, 6, 192] | float32 | streaming state |
| `block_0_prev` .. `block_7_prev` (IN) | [1, 6, 384] each | float32 | streaming state per block |

#### Outputs

| Tensor | Shape | Type | Note |
|--------|-------|------|------|
| `Identity` (OUT 0) | [1, 16, 601] | float32 | x_real spectrum NTK |
| `Identity_1` (OUT 1) | [1, 16, 601] | float32 | x_imag spectrum NTK |
| `Identity_2` (OUT 2) | [1, 6, 192] | float32 | new embed_prev |
| `Identity_3..10` (OUT 3..10) | [1, 384, 6] each | float32 | new block_N_prev (**note: NCT not NTC!**) |

> ⚠ State outputs `Identity_3..10` have shape **[1, 384, 6] NCT** but inputs `block_N_prev` expect **[1, 6, 384] NTC**. Transpose before feeding back.

#### IRFFT (Kotlin)

The model outputs `x_real[1, F, K]` and `x_imag[1, F, K]` where:
- `F = VOCOS_CHUNK_FRAMES = 16` (frames)
- `K = 601 = VOCOS_N_FFT/2 + 1` (complex spectrum bins)

Each of the `F` frames is a half-spectrum of a real IRFFT with `n = VOCOS_N_FFT = 1200`.

**Recommended library**: [JTransforms](https://github.com/wendykierp/JTransforms) (pure Java, no NDK required, well-tested on Android).

```gradle
// build.gradle.kts
implementation("com.github.wendykierp:JTransforms:3.1")
```

```kotlin
import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D

val fft = FloatFFT_1D(VOCOS_N_FFT.toLong())

/**
 * IRFFT of a single frame.
 *
 * @param real  FloatArray of length K = N/2+1
 * @param imag  FloatArray of length K = N/2+1
 * @param n     FFT size (= VOCOS_N_FFT = 1200)
 * @return      FloatArray of length n (time-domain frame)
 */
fun irfft(real: FloatArray, imag: FloatArray, n: Int): FloatArray {
    // JTransforms expects interleaved complex [re0, im0, re1, im1, ...]
    // Full two-sided spectrum for real IFFT has length n, but JTransforms realInverse
    // expects the compact half-complex format.
    val packed = FloatArray(n)
    // Pack into JTransforms half-complex format:
    // packed[0] = re[0], packed[1] = re[n/2], packed[2k] = re[k], packed[2k+1] = im[k]  (k=1..n/2-1)
    packed[0] = real[0]
    if (n % 2 == 0) packed[1] = real[n / 2]
    for (k in 1 until n / 2) {
        packed[2 * k]     = real[k]
        packed[2 * k + 1] = imag[k]
    }
    fft.realInverse(packed, true)  // true = scale by 1/n
    return packed
}
```

#### Overlap-Add (OLA)

Each IRFFT frame has `VOCOS_N_FFT = 1200` samples with a hop of `VOCOS_HOP = 300`.
`VOCOS_TAIL = VOCOS_N_FFT - VOCOS_HOP = 900` samples overlap between frames.

```kotlin
/**
 * Overlap-add for a batch of time-domain frames.
 *
 * @param frames    FloatArray [F * N_FFT], F frames stacked (each of length N_FFT)
 * @param prevTail  FloatArray of length TAIL, carry-over from previous call (init to zeros)
 * @param window    FloatArray of length N_FFT, Hann window
 * @return Pair(audio [F * HOP], newTail [TAIL])
 */
fun overlapAdd(
    frames: FloatArray,
    prevTail: FloatArray,
    window: FloatArray
): Pair<FloatArray, FloatArray> {
    val nFrames = frames.size / VOCOS_N_FFT
    val audio = FloatArray(nFrames * VOCOS_HOP)
    val tail = prevTail.copyOf()  // [TAIL]

    for (f in 0 until nFrames) {
        val fOffset = f * VOCOS_N_FFT
        // Apply Hann window
        val windowed = FloatArray(VOCOS_N_FFT) { i -> frames[fOffset + i] * window[i] }

        // Output sample buffer for this frame
        val buf = FloatArray(VOCOS_N_FFT)
        // Add overlap from previous tail
        for (i in 0 until VOCOS_TAIL) buf[i] = tail[i]
        // Add current windowed frame
        for (i in 0 until VOCOS_N_FFT) buf[i] += windowed[i]

        // First HOP samples go to output
        for (i in 0 until VOCOS_HOP) audio[f * VOCOS_HOP + i] = buf[i]
        // Remaining TAIL samples become new tail
        buf.copyInto(tail, 0, VOCOS_HOP, VOCOS_N_FFT)
    }
    return Pair(audio, tail)
}
```

#### Hann Window

```kotlin
val hannWindow = FloatArray(VOCOS_N_FFT) { i ->
    (0.5f * (1f - cos(2f * PI.toFloat() * i / VOCOS_N_FFT)))
}
```

#### Full S7 Loop

```kotlin
// Initialize state (all zeros)
var embedPrev    = FloatArray(6 * 192)
val blockPrevs   = Array(8) { FloatArray(6 * 384) }
var prevTail     = FloatArray(VOCOS_TAIL)

val audioChunks  = mutableListOf<FloatArray>()
val totalFrames  = tF0Actual
var pos = 0

while (pos < totalFrames) {
    val end   = minOf(totalFrames, pos + VOCOS_CHUNK_FRAMES)
    val valid = end - pos

    // Slice conditioned[192, T_F0] NCT → chunk [16, 192] NTC (pad if needed)
    val chunk = FloatArray(VOCOS_CHUNK_FRAMES * 192)
    for (f in 0 until valid) {
        for (c in 0 until 192) {
            chunk[f * 192 + c] = conditioned[c * T_F0 + (pos + f)]
        }
    }
    // Padding frames (if valid < CHUNK_FRAMES) remain zero

    // Run model
    vocosInterp.setInputTensor("conditioned_chunk", chunk)   // [1,16,192] NTC
    vocosInterp.setInputTensorFlat("embed_prev", embedPrev)  // [1,6,192]
    for (b in 0..7) vocosInterp.setInputTensorFlat("block_${b}_prev", blockPrevs[b])  // [1,6,384]
    vocosInterp.run()

    val xReal = vocosInterp.getOutputByIndex(0)  // [1,16,601] NTK (x_real)
    val xImag = vocosInterp.getOutputByIndex(1)  // [1,16,601] (x_imag)
    // Update state
    embedPrev = vocosInterp.getOutputByIndex(2)  // [1,6,192]
    for (b in 0..7) {
        val rawState = vocosInterp.getOutputByIndex(3 + b)  // [1,384,6] NCT
        blockPrevs[b] = transposeNCTtoNTC(rawState, 384, 6) // → [1,6,384] NTC
    }

    // IRFFT: [16, 601] → [16, 1200]
    val timeFrames = FloatArray(VOCOS_CHUNK_FRAMES * VOCOS_N_FFT)
    for (f in 0 until VOCOS_CHUNK_FRAMES) {
        val real = FloatArray(VOCOS_IRNC_BINS) { k -> xReal[f * VOCOS_IRNC_BINS + k] }
        val imag = FloatArray(VOCOS_IRNC_BINS) { k -> xImag[f * VOCOS_IRNC_BINS + k] }
        val frame = irfft(real, imag, VOCOS_N_FFT)
        frame.copyInto(timeFrames, f * VOCOS_N_FFT)
    }

    // Overlap-add
    val (audioSegment, newTail) = overlapAdd(timeFrames, prevTail, hannWindow)
    prevTail = newTail

    // Only keep valid samples (discard padding frames)
    audioChunks.add(audioSegment.copyOfRange(0, valid * VOCOS_HOP))
    pos = end
}

val audio = FloatArray(audioChunks.sumOf { it.size })
var offset = 0
for (chunk in audioChunks) { chunk.copyInto(audio, offset); offset += chunk.size }
```

---

## 6. Tensor Layout Reference

| Model | Input layout | Output layout |
|-------|-------------|---------------|
| BERT | `[1, T]` | `[1, T, C]` **NTC** |
| Duration predictor | `d_en [1, T, C]` **NTC** | `d_enc [1, C, T]` **NCT**, `pred_dur [T]`, `t_en [1, C, T]` **NCT** |
| Acoustic expand | `[1, T, C]` **NTC** | `[1, T, C]` **NTC** |
| F0N predictor | `en [1, C, T]` **NCT** | `F0 [1, T]`, `N [1, T]` |
| Conditioner | `[1, T, 642]` **NTC** | `[1, C, T]` **NCT** |
| Vocos stream chunk | `chunk [1, F, C]` **NTK** | `x_real [1, F, K]`, `x_imag [1, F, K]` **NTK**; state: Identity_2 **NTC**, Identity_3..10 **NCT** |

---

## 7. LiteRT Setup (Android)

```gradle
// build.gradle.kts
implementation("com.google.ai.edge.litert:litert:1.0.1")
implementation("com.google.ai.edge.litert:litert-gpu:1.0.1")
```

### CPU Delegate (baseline)

```kotlin
import com.google.ai.edge.litert.Interpreter

val options = Interpreter.Options().apply {
    setNumThreads(4)
}
val interp = Interpreter(modelFile, options)
```

### GPU Delegate

```kotlin
import com.google.ai.edge.litert.gpu.CompatibilityList
import com.google.ai.edge.litert.gpu.GpuDelegate

val compatList = CompatibilityList()
val options = Interpreter.Options()
if (compatList.isDelegateSupportedOnThisDevice) {
    val gpuOptions = compatList.bestOptionsForThisDevice
    options.addDelegate(GpuDelegate(gpuOptions))
} else {
    options.setNumThreads(4)
}
val interp = Interpreter(modelFile, options)
```

> **GPU delegate note**: The GPU delegate requires float32 models. Do **not** use float16 `.tflite` files with the GPU delegate — use the float32 variants and let the delegate handle precision internally.

### NNAPI Delegate (NPU/DSP — requires API 27+)

```kotlin
import com.google.ai.edge.litert.NnApiDelegate

val nnApiOptions = NnApiDelegate.Options().apply {
    setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
    setModelToken("kokoro_bert")  // unique per model for caching
}
options.addDelegate(NnApiDelegate(nnApiOptions))
```

### Pixel 10 Recommendation

The **Pixel 10** features the **Tensor G4** SoC with a dedicated NPU (Tensor Processing Unit). Recommended configuration:
1. Try **NNAPI delegate** first — best for Tensor G4 NPU acceleration.
2. Fall back to **GPU delegate** — good for conv-heavy models (conditioner, acoustic_expand).
3. Fall back to **CPU** (4 threads) as baseline.

Performance-critical models for NPU: `bert`, `duration_predictor`, `f0n_predictor`.
Conditioner and acoustic_expand are small and fast on GPU.

```kotlin
// Recommended Pixel 10 setup
fun createInterpreter(modelFile: File, modelToken: String): Interpreter {
    val options = Interpreter.Options()
    try {
        val nnApi = NnApiDelegate(NnApiDelegate.Options().apply {
            setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
            setModelToken(modelToken)
        })
        options.addDelegate(nnApi)
    } catch (e: Exception) {
        val compatList = CompatibilityList()
        if (compatList.isDelegateSupportedOnThisDevice) {
            options.addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
        } else {
            options.setNumThreads(4)
        }
    }
    return Interpreter(modelFile, options)
}
```

---

## 8. Quantization

### Float16 (recommended for Android deployment)

Float16 (FP16) variants are available for all models:

```
bert_float16.tflite                  ~13 MB
duration_predictor_float16.tflite    ~27 MB
acoustic_expand_float16.tflite       ~3.7 MB
f0n_predictor_float16.tflite         ~13.5 MB
vocoder_conditioner_float16.tflite   ~0.6 MB
vocoder_stream_chunk_float16.tflite  ~16.3 MB
```

For FP16 models, set inputs as float32 (they are auto-quantized internally by LiteRT).

### Int8 Dynamic Range Quantization

For maximum compression, int8 post-training quantization can reduce models by ~4×.
However, LSTM-heavy models (bert, duration predictor, f0n, acoustic_expand) may suffer perceptible quality degradation. Recommended int8 candidates:
- `vocoder_conditioner` (small, conv-based) ✓
- `vocoder_stream_chunk` (streaming conv backbone) ✓ with calibration

BERT, duration predictor, and f0n predictor should remain FP32 or FP16 for quality.

---

## 9. DFT Library Recommendation

For implementing IRFFT on **Android Pixel 10** (ARMv9 with Tensor G4):

| Library | Language | NDK required | FFT size 1200 perf | Recommendation |
|---------|----------|-------------|---------------------|----------------|
| **JTransforms 3.1** | Pure Java | No | ~50 µs/frame | ✅ **Best for pure Kotlin** |
| PFFFT (via NDK) | C | Yes | ~5 µs/frame | ✅ Best raw performance |
| KISSFFT (via NDK) | C | Yes | ~10 µs/frame | Good NDK alternative |
| Apache Commons Math | Pure Java | No | ~200 µs/frame | Too slow |

**Recommendation**: Use **JTransforms** for a pure-Kotlin solution. It uses SIMD-friendly vectorized operations and provides `FloatFFT_1D.realInverse()` which directly computes the N=1200 IRFFT.

For production with NDK: **PFFFT** (Powerful Fast FFT) is the fastest on ARM. The 16 IRFFT calls per chunk at 300 samples/frame means 4800 samples (0.2s) per chunk, easily real-time.

---

## 10. Complete Inference Pipeline (Kotlin)

```kotlin
class KokoroTts(context: Context, voiceFile: File) {
    // Models
    private val bertInterp     = createInterpreter(File(context.filesDir, "bert_float32.tflite"), "bert")
    private val durInterp      = createInterpreter(File(context.filesDir, "duration_predictor_float32.tflite"), "dur")
    private val acexpInterp    = createInterpreter(File(context.filesDir, "acoustic_expand_float32.tflite"), "acexp")
    private val f0nInterp      = createInterpreter(File(context.filesDir, "f0n_predictor_float32.tflite"), "f0n")
    private val condInterp     = createInterpreter(File(context.filesDir, "vocoder_conditioner_float32.tflite"), "cond")
    private val vocosInterp    = createInterpreter(File(context.filesDir, "vocoder_stream_chunk_float32.tflite"), "vocos")

    private val voicePack      = loadVoicePack(voiceFile)         // [N_phonemes, 256]
    private val hannWindow     = buildHannWindow(VOCOS_N_FFT)

    fun generate(text: String, speed: Float = 1.0f): FloatArray {
        val (phonemes, rawIds) = g2p(text)
        val style = voicePack[phonemes.size - 1]

        // Pad input
        val inputIds = IntArray(MAX_INPUT_LENGTH).also { arr ->
            arr[0] = 0
            rawIds.forEachIndexed { i, id -> arr[i + 1] = id }
            arr[rawIds.size + 1] = 0
        }
        val textMask = FloatArray(MAX_INPUT_LENGTH).also { arr ->
            for (i in 0..rawIds.size + 1) arr[i] = 1.0f
        }

        // S1: BERT
        val dEn = runBert(inputIds, textMask)

        // S2: Duration Predictor
        val (predDur, dEnc, tEnStatic) = runDuration(dEn, style, textMask, speed, inputIds)

        // S3: Expand durations
        val (expandedIdx, tAcoustic) = expandDurations(predDur)
        val dEncExp = indexSelectAndPad(dEnc, expandedIdx, tAcoustic, 640, T_ACOUSTIC)
        val asr     = indexSelectNCT(tEnStatic, expandedIdx, tAcoustic, 512)

        // S4: Acoustic expand
        val enNtc = runAcousticExpand(dEncExp)           // [T_ACOUSTIC, 512] NTC
        val enNct = transposeNTCtoNCT(enNtc, T_ACOUSTIC, 512)  // [512, T_ACOUSTIC] NCT

        // S5: F0/N predictor
        val (f0Full, nFull) = runF0N(enNct, style)
        val tF0 = 2 * tAcoustic
        val f0 = f0Full.copyOfRange(0, tF0)
        val n  = nFull.copyOfRange(0, tF0)

        // S6: Conditioner
        val features    = buildVocosFeatures(asr, f0, n, style, tAcoustic, tF0)
        val conditioned = runConditioner(features)       // [192, T_F0] NCT

        // S7: Vocos streaming IRFFT + OLA
        return runVocos(conditioned, tF0)
    }
}
```

---

## 11. Notes on Padding and Quality

The TFLite models use **static shapes** (all padded to max sizes). This means:
- Short texts (< 50 phonemes) see significant zero-padding in BiLSTM models.
- The BiLSTM backward pass starts from the padded (zero) positions, affecting actual-position outputs.
- Audio quality improves with longer input texts where padding proportion is smaller.
- Expected WER for short sentences may be higher than the PyTorch reference.

For best quality on Android, prefer processing paragraphs over individual short sentences.
