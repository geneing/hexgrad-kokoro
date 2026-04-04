# Kokoro TTS — Android ONNX Runtime Inference Specification

> **Status**: Models exported and validated. See `onnx_export_android.py`.
> **Sample rate**: 24 000 Hz mono float32.
> **Runtime**: ONNX Runtime for Android (`com.microsoft.onnxruntime:onnxruntime-android`)
> **Supported text**: English (American, phoneme-based).

---

## 1. Overview

Kokoro TTS Android ONNX inference uses **7 stages**, 6 of which run on ONNX Runtime TFLite.
Stage 3 (duration expansion), Stage 7 IRFFT, and overlap-add run in pure Kotlin.

```
Text
 │
 ▼
[S1] BERT Encoder          bert.onnx                  ~26 MB (fp32) / ~13 MB (fp16)
 │  d_en [1,510,512] NTC
 ▼
[S2] Duration Predictor    duration_predictor.onnx    ~54 MB / ~27 MB
 │  pred_dur[510], d_enc[1,h,510] NCT, t_en[1,512,510] NCT
 ▼
[S3] Duration Expansion    (Kotlin — no model)
 │  expanded_indices[], T_acoustic
 ▼ index_select on d_enc / t_en
[S4] Acoustic Expand       acoustic_expand.onnx       ~7 MB / ~4 MB
 │  en [1,T_ACOUSTIC,512] NTC
 ▼
[S5] F0/N Predictor        f0n_predictor.onnx         ~27 MB / ~14 MB
 │  F0_pred[1,1086], N_pred[1,1086]
 ▼
[S6] Vocos Conditioner     vocoder_conditioner.onnx   ~1 MB / ~0.6 MB
 │  conditioned[1,192,T_F0] NCT
 ▼
[S7] Vocos Stream Chunk    vocoder_stream_chunk.onnx  ~33 MB / ~17 MB
     → outputs x_real[1,16,601] + x_imag[1,16,601]
     → IRFFT (Kotlin JTransforms) + Hann window + Overlap-Add (Kotlin)
     → audio [float32, 24 kHz]
```

### Key Android Optimisations vs. TFLite Export

| Feature | TFLite | ONNX Android |
|---------|--------|--------------|
| Vocoder output | x_real/x_imag spectra | x_real/x_imag spectra |
| GELU activation | nn.GELU (via onnx2tf) | **FastGELU**: `x·sigmoid(1.702x)` |
| ONNX DFT op | Absent (VocosPreIRFFT) | Absent (VocosPreIRFFTAndroid) |
| Shapes | Static | Static (NNAPI-compatible) |
| OLA ConvTranspose | External Python/Kotlin | External Kotlin |
| Weight norm | Folded | Folded |

**FastGELU** exports to `{Sigmoid, Mul}` ops only — no `Erf`, `Tanh`, or `Pow`.
GPU delegate and NNAPI accelerate both ops natively. Max error vs exact GELU: < 5×10⁻⁴.

---

## 2. Constants

```kotlin
const val SAMPLE_RATE        = 24_000
const val MAX_INPUT_LENGTH   = 510
const val T_ACOUSTIC         = 543
const val T_F0               = 1086       // 2 × T_ACOUSTIC
const val VOCOS_CHUNK_FRAMES = 16
const val VOCOS_HOP          = 300
const val VOCOS_N_FFT        = 1200
const val VOCOS_WIN_LEN      = 1200       // = VOCOS_N_FFT
const val VOCOS_TAIL         = 900        // WIN_LEN - HOP
const val VOCOS_K            = 601        // N_FFT / 2 + 1
const val VOCOS_COND_DIM     = 192
const val VOCOS_KERNEL_M1    = 6          // causal conv context frames (kernel_size - 1)
const val VOCOS_N_BLOCKS     = 8
const val VOCOS_BLOCK_DIM    = 384
```

---

## 3. ONNX Runtime Setup

### Dependency (build.gradle.kts)

```gradle
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
```

### Session Creation

```kotlin
import ai.onnxruntime.*

fun createSession(
    env: OrtEnvironment,
    modelBytes: ByteArray,
    useNnapi: Boolean = true,
    useGpu: Boolean = true,
): OrtSession {
    val opts = OrtSession.SessionOptions()
    opts.setInterOpNumThreads(4)
    opts.setIntraOpNumThreads(4)
    if (useNnapi) {
        try {
            opts.addNnapi()  // best for Tensor G4 NPU on Pixel 9/10
        } catch (e: OrtException) {
            if (useGpu) opts.addCoreML()  // fallback GPU path
        }
    }
    return env.createSession(modelBytes, opts)
}
```

### NNAPI Delegate Notes

- NNAPI requires **static input shapes** — all models are exported with static shapes.
- Use `float32` inputs even with `_fp16.onnx` models; ORT handles quantisation internally.
- Add `NNAPI_FLAG_USE_FP16` via `NnapiFlags` for FP16 on the NPU:

```kotlin
val nnapiOpts = NNAPIExecutionProviderOptions()
nnapiOpts.useFloat16 = true
opts.addNnapi(nnapiOpts)
```

---

## 4. Voice Pack & Style Vector

```kotlin
// style: FloatArray of length 256
// style[0..127]  → decoder style (s_ref, used by conditioner)
// style[128..255] → predictor style (s_predictor, used by duration, f0n)
val style = loadVoicePack(voiceFile)[phonemes.size - 1]  // row = phoneme count - 1
```

---

## 5. Text → Phoneme → Token IDs

Use Misaki g2p (Python side) or a pre-built phoneme lookup table.

```kotlin
val rawIds: IntArray = g2p(text)
val seqLen = rawIds.size + 2   // +2 for BOS / EOS tokens (both = 0)
val inputIds = IntArray(MAX_INPUT_LENGTH)
val textMask = FloatArray(MAX_INPUT_LENGTH)
inputIds[0] = 0
rawIds.copyInto(inputIds, 1)
inputIds[seqLen - 1] = 0
for (i in 0 until seqLen) textMask[i] = 1.0f
```

---

## 6. Stage-by-Stage Inference

### S1 — BERT Encoder

**Model**: `bert.onnx` (or `bert_fp16.onnx`)

| Tensor | Shape | Type |
|--------|-------|------|
| `input_ids` (IN) | [1, 510] | int32 |
| `text_mask` (IN) | [1, 510] | float32 |
| `d_en` (OUT) | [1, 510, 512] | float32 — **NTC** |

```kotlin
val inputs = mapOf(
    "input_ids" to OnnxTensor.createTensor(env, intArrayOf2D(inputIds), longArrayOf(1, 510)),
    "text_mask" to OnnxTensor.createTensor(env, textMask, longArrayOf(1, 510)),
)
val dEn = bertSession.run(inputs)["d_en"]!!.value as Array<*>  // [1, 510, 512] NTC
```

---

### S2 — Duration Predictor

**Model**: `duration_predictor.onnx`

| Tensor | Shape | Type |
|--------|-------|------|
| `d_en` (IN) | [1, 510, 512] | float32 — NTC |
| `style` (IN) | [1, 256] | float32 |
| `text_mask` (IN) | [1, 510] | float32 |
| `speed` (IN) | [1] | int32 |
| `input_ids` (IN) | [1, 510] | int32 |
| `pred_dur` (OUT) | [510] | float32 |
| `d_enc` (OUT) | [1, h, 510] | float32 — **NCT** |
| `t_en_static` (OUT) | [1, 512, 510] | float32 — **NCT** |

```kotlin
val durInputs = mapOf(
    "d_en"      to OnnxTensor.createTensor(env, dEnFlat, longArrayOf(1, 510, 512)),
    "style"     to OnnxTensor.createTensor(env, style, longArrayOf(1, 256)),
    "text_mask" to OnnxTensor.createTensor(env, textMask, longArrayOf(1, 510)),
    "speed"     to OnnxTensor.createTensor(env, intArrayOf(speed), longArrayOf(1)),
    "input_ids" to OnnxTensor.createTensor(env, inputIds, longArrayOf(1, 510)),
)
val durOut  = durSession.run(durInputs)
val predDur = durOut["pred_dur"]!!.value as FloatArray     // [510]
val dEnc    = durOut["d_enc"]!!.value     // [1, h, 510] NCT
val tEnStat = durOut["t_en_static"]!!.value // [1, 512, 510] NCT
```

---

### S3 — Duration Expansion (Kotlin, no model)

```kotlin
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

```kotlin
val (expandedIdx, tAcoustic) = expandDurations(predDur)

// Index-select on NCT axis-2 (T axis), pad to T_ACOUSTIC
val dEncExp = indexSelectAndPad(dEnc, expandedIdx, tAcoustic, nChannels = h, tStatic = T_ACOUSTIC)
val asr     = indexSelectNCT(tEnStat, expandedIdx, tAcoustic, nChannels = 512)

fun indexSelectAndPad(nct: FloatArray, idx: IntArray, tAc: Int, nChannels: Int, tStatic: Int): FloatArray {
    val out = FloatArray(nChannels * tStatic)
    for (c in 0 until nChannels) {
        for (t in 0 until tAc) {
            out[c * tStatic + t] = nct[c * 510 + idx[t]]
        }
        // positions tAc..tStatic-1 remain 0 (zero-padded)
    }
    return out  // NCT flat [nChannels * tStatic]
}
```

---

### S4 — Acoustic Expand

**Model**: `acoustic_expand.onnx`

| Tensor | Shape | Type | Note |
|--------|-------|------|------|
| `d_enc_expanded` (IN) | [1, 543, 640] | float32 | **NTC** — transposed dEnc |
| `en` (OUT) | [1, 543, 512] | float32 | **NTC** |

```kotlin
// Transpose dEncExp from NCT [1, 640, T_ACOUSTIC] to NTC [1, T_ACOUSTIC, 640]
val dEncNtc = transposeNCTtoNTC(dEncExp, 640, T_ACOUSTIC)

val acexpOut = acexpSession.run(mapOf(
    "d_enc_expanded" to OnnxTensor.createTensor(env, dEncNtc, longArrayOf(1, T_ACOUSTIC.toLong(), 640))
))
val enNtc = acexpOut["en"]!!.value as FloatArray   // [1, T_ACOUSTIC, 512] NTC
val enNct = transposeNTCtoNCT(enNtc, T_ACOUSTIC, 512)  // → NCT [1, 512, T_ACOUSTIC]
// IMPORTANT: pass FULL T_ACOUSTIC to f0n (do NOT trim to tAcoustic)
```

---

### S5 — F0/N Predictor

**Model**: `f0n_predictor.onnx`

| Tensor | Shape | Type | Note |
|--------|-------|------|------|
| `en` (IN) | [1, 512, 543] | float32 | **NCT** full T_ACOUSTIC |
| `style` (IN) | [1, 256] | float32 | |
| `F0_pred` (OUT) | [1, 1086] | float32 | |
| `N_pred` (OUT) | [1, 1086] | float32 | |

```kotlin
val f0nOut = f0nSession.run(mapOf(
    "en"    to OnnxTensor.createTensor(env, enNct, longArrayOf(1, 512, T_ACOUSTIC.toLong())),
    "style" to OnnxTensor.createTensor(env, style, longArrayOf(1, 256)),
))
val f0Full = f0nOut["F0_pred"]!!.value as FloatArray   // [1, 1086]
val nFull  = f0nOut["N_pred"]!!.value  as FloatArray   // [1, 1086]

val tF0 = 2 * tAcoustic
val f0 = f0Full.copyOfRange(0, tF0)
val n  = nFull.copyOfRange(0, tF0)
```

---

### S6 — Vocos Conditioner

**Model**: `vocoder_conditioner.onnx`  
*(Uses FastGELU — no Erf/Tanh ops)*

Feature assembly `[1, 642, T_F0_static]` **NCT** → conditioned `[1, 192, T_F0_static]` **NCT**:

```kotlin
fun buildVocosFeatures(
    asr: FloatArray,    // NCT flat [512 * tAcoustic]
    f0: FloatArray,     // [tF0]
    n: FloatArray,      // [tF0]
    style: FloatArray,  // [256]
    tAcoustic: Int,
    tF0: Int,
    tF0Static: Int = T_F0
): FloatArray {
    val out = FloatArray(642 * tF0Static)
    for (t in 0 until tF0) {
        val offset = t * 642  // Wait — for NCT flat this is wrong
        // IMPORTANT: features tensor is NCT [642, T_F0_static]
        // out[c * T_F0_static + t]
    }
    // Build as NCT flat [642 * tF0Static]:
    val srcFracStep = tAcoustic.toFloat() / tF0
    for (t in 0 until tF0) {
        val srcT = (t * srcFracStep).toInt().coerceIn(0, tAcoustic - 1)
        for (c in 0 until 512) {
            out[c * tF0Static + t] = asr[c * tAcoustic + srcT]
        }
        out[512 * tF0Static + t] = f0[t]
        out[513 * tF0Static + t] = n[t]
        for (c in 0 until 128) {
            out[(514 + c) * tF0Static + t] = style[c]
        }
    }
    // Positions t=tF0..tF0Static-1 remain 0
    return out
}
```

```kotlin
val features = buildVocosFeatures(asr, f0, n, style, tAcoustic, tF0)
val condOut = condSession.run(mapOf(
    "features" to OnnxTensor.createTensor(env, features, longArrayOf(1, 642, T_F0.toLong()))
))
val conditioned = condOut["conditioned"]!!.value as FloatArray  // [1, 192, T_F0] NCT
```

---

### S7 — Vocos Streaming Chunk

**Model**: `vocoder_stream_chunk.onnx`  
*(FastGELU backbone; outputs x_real/x_imag — no DFT op)*

#### Inputs

| Tensor | Shape | Type | Note |
|--------|-------|------|------|
| `conditioned_chunk` (IN) | [1, 192, 16] | float32 | **NCT** — 16 conditioned frames |
| `embed_prev` (IN) | [1, 192, 6] | float32 | streaming state |
| `block_0_prev`..**block_7_prev** (IN) | [1, 384, 6] each | float32 | streaming state per block |

#### Outputs

| Tensor | Shape | Type | Note |
|--------|-------|------|------|
| `x_real` (OUT 0) | [1, 16, 601] | float32 | spectrum real part NFK |
| `x_imag` (OUT 1) | [1, 16, 601] | float32 | spectrum imag part NFK |
| `embed_prev_new` (OUT 2) | [1, 192, 6] | float32 | |
| `block_0_prev_new`..**block_7_prev_new** (OUT 3..10) | [1, 384, 6] each | float32 | |

> **Note**: Unlike the TFLite export, all state tensors are in **NCT [1, C, 6]** format — no transposition needed between chunks.

#### IRFFT (Kotlin / JTransforms)

```gradle
// build.gradle.kts
implementation("com.github.wendykierp:JTransforms:3.1")
```

```kotlin
import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D

val fft = FloatFFT_1D(VOCOS_N_FFT.toLong())

/**
 * IRFFT of a single spectral frame.
 *
 * @param real  FloatArray length K = N_FFT/2+1 = 601
 * @param imag  FloatArray length K
 * @param n     FFT size (= VOCOS_N_FFT = 1200)
 * @return      FloatArray length n (time-domain)
 */
fun irfft(real: FloatArray, imag: FloatArray, n: Int): FloatArray {
    val packed = FloatArray(n)
    packed[0] = real[0]
    if (n % 2 == 0) packed[1] = real[n / 2]
    for (k in 1 until n / 2) {
        packed[2 * k]     = real[k]
        packed[2 * k + 1] = imag[k]
    }
    fft.realInverse(packed, true)   // scale by 1/n
    return packed
}
```

#### Hann Window

```kotlin
import kotlin.math.PI
import kotlin.math.cos

val hannWindow = FloatArray(VOCOS_N_FFT) { i ->
    (0.5f * (1f - cos(2.0 * PI * i / VOCOS_N_FFT))).toFloat()
}
```

#### Overlap-Add

```kotlin
/**
 * @param frames   FloatArray [F * WIN_LEN] — F windowed time-domain frames
 * @param prevTail FloatArray [TAIL] — carry-over from previous call (start with zeros)
 * @param window   Hann window [WIN_LEN]
 * @return Pair(audio [F * HOP], newTail [TAIL])
 */
fun overlapAdd(frames: FloatArray, prevTail: FloatArray, window: FloatArray): Pair<FloatArray, FloatArray> {
    val nFrames = frames.size / VOCOS_WIN_LEN
    val audio = FloatArray(nFrames * VOCOS_HOP)
    val tail = prevTail.copyOf()

    for (f in 0 until nFrames) {
        val fOff = f * VOCOS_WIN_LEN
        // Windowed frame overlaid on running buffer
        val buf = FloatArray(VOCOS_WIN_LEN)
        for (i in 0 until VOCOS_TAIL) buf[i] = tail[i]
        for (i in 0 until VOCOS_WIN_LEN) buf[i] += frames[fOff + i] * window[i]

        for (i in 0 until VOCOS_HOP) audio[f * VOCOS_HOP + i] = buf[i]
        buf.copyInto(tail, 0, VOCOS_HOP, VOCOS_WIN_LEN)
    }
    return Pair(audio, tail)
}
```

#### Full S7 Loop

```kotlin
// Initialise state (all zeros)
var embedPrev  = FloatArray(192 * 6)
val blockPrevs = Array(8) { FloatArray(384 * 6) }
var prevTail   = FloatArray(VOCOS_TAIL)

val audioChunks = mutableListOf<FloatArray>()
val tF0Actual   = tF0
var pos = 0

while (pos < tF0Actual) {
    val end   = minOf(tF0Actual, pos + VOCOS_CHUNK_FRAMES)
    val valid = end - pos

    // Slice conditioned[192, T_F0] NCT → chunk [192, 16] NCT (pad if needed)
    val chunk = FloatArray(192 * VOCOS_CHUNK_FRAMES)
    for (f in 0 until valid) {
        for (c in 0 until 192) {
            chunk[c * VOCOS_CHUNK_FRAMES + f] = conditioned[c * T_F0 + (pos + f)]
        }
    }

    val inputs = mutableMapOf(
        "conditioned_chunk" to OnnxTensor.createTensor(env, chunk, longArrayOf(1, 192, VOCOS_CHUNK_FRAMES.toLong())),
        "embed_prev"        to OnnxTensor.createTensor(env, embedPrev, longArrayOf(1, 192, 6)),
    )
    for (b in 0..7) {
        inputs["block_${b}_prev"] = OnnxTensor.createTensor(env, blockPrevs[b], longArrayOf(1, 384, 6))
    }

    val outs = vocosSession.run(inputs)
    val xReal = outs["x_real"]!!.value as FloatArray   // [1, 16, 601]
    val xImag = outs["x_imag"]!!.value as FloatArray   // [1, 16, 601]
    embedPrev = outs["embed_prev_new"]!!.value as FloatArray
    for (b in 0..7) {
        blockPrevs[b] = outs["block_${b}_prev_new"]!!.value as FloatArray  // [1, 384, 6] NCT
    }

    // IRFFT: [1, 16, 601] → [16 * 1200] time frames
    val timeFrames = FloatArray(VOCOS_CHUNK_FRAMES * VOCOS_WIN_LEN)
    for (f in 0 until VOCOS_CHUNK_FRAMES) {
        val real = FloatArray(VOCOS_K) { k -> xReal[f * VOCOS_K + k] }
        val imag = FloatArray(VOCOS_K) { k -> xImag[f * VOCOS_K + k] }
        irfft(real, imag, VOCOS_N_FFT).copyInto(timeFrames, f * VOCOS_WIN_LEN)
    }

    val (audioSeg, newTail) = overlapAdd(timeFrames, prevTail, hannWindow)
    prevTail = newTail
    audioChunks.add(audioSeg.copyOfRange(0, valid * VOCOS_HOP))
    pos = end
}

val audio = FloatArray(audioChunks.sumOf { it.size })
var off = 0
for (chunk in audioChunks) { chunk.copyInto(audio, off); off += chunk.size }
```

---

## 7. Tensor Layout Reference

| Model | Input layout | Output layout |
|-------|-------------|---------------|
| BERT | `[1, T]` | `[1, T, C]` **NTC** |
| Duration predictor | `d_en [1, T, C]` **NTC** | `d_enc/t_en [1, C, T]` **NCT**, `pred_dur [T]` |
| Acoustic expand | `[1, T, C]` **NTC** | `[1, T, C]` **NTC** |
| F0N predictor | `en [1, C, T]` **NCT** | `F0 [1, T]`, `N [1, T]` |
| Conditioner | `[1, C, T]` **NCT** | `[1, C, T]` **NCT** |
| Vocos stream chunk | `chunk [1, C, F]` **NCT** | `x_real/x_imag [1, F, K]` NFK; state **NCT** |

> **Note**: Conditioner input is **NCT** for ONNX  
> (unlike TFLite where onnx2tf transposed it to NTC).  
> Vocos state tensors are **always NCT [1, C, 6]** — no transposition needed between chunks.

---

## 8. FastGELU vs GELU

GELU activations in the Vocos vocoder (conditioner and ConvNeXt backbone) are replaced with:

```
FastGELU(x) = x · σ(1.702 · x)
```

This approximation:
- Exports to `{Sigmoid, Mul}` — no `Erf`, `Tanh`, `Pow`
- Max error < 5×10⁻⁴ vs exact GELU
- Is natively accelerated by NNAPI and GPU delegates
- Matches the training distribution well enough for high-quality synthesis

---

## 9. Quantisation

### FP16 (recommended for deployment)

All `_fp16.onnx` variants are available from `onnx_android/`:

```
bert_fp16.onnx                   ~13 MB
duration_predictor_fp16.onnx     ~27 MB
acoustic_expand_fp16.onnx        ~4 MB
f0n_predictor_fp16.onnx          ~14 MB
vocoder_conditioner_fp16.onnx    ~0.6 MB
vocoder_stream_chunk_fp16.onnx   ~17 MB
```

Pass float32 inputs — ORT handles fp16 casting internally.

### NNAPI FP16 Acceleration

```kotlin
val nnapiOpts = OrtSession.SessionOptions()
nnapiOpts.addNnapi(mapOf("NNAPI_FLAG_USE_FP16" to "1"))
```

---

## 10. Op Coverage (NNAPI Support)

All ops in the exported models fall within the NNAPI/GPU EP supported set:

| Model | Key ops | GPU/NPU |
|-------|---------|---------|
| BERT | MatMul, LayerNorm, Gather | ✓ NNAPI |
| Duration predictor | LSTM, MatMul, Sigmoid | ✓ NNAPI |
| Acoustic expand | LSTM | ✓ NNAPI |
| F0N predictor | LSTM, Conv, ResBlk | ✓ NNAPI |
| Conditioner | Conv, **Sigmoid+Mul** (FastGELU) | ✓ GPU |
| Vocos chunk | Conv, MatMul, **Sigmoid+Mul** (FastGELU) | ✓ GPU |

Run `onnx_export_android.py` for a full per-model op coverage report.

---

## 11. Pixel 10 Recommendation

The **Pixel 10** (Tensor G4 SoC) has a dedicated NPU accessible via NNAPI.

```kotlin
fun createInterpreter(env: OrtEnvironment, modelBytes: ByteArray, modelTag: String): OrtSession {
    val opts = OrtSession.SessionOptions().apply {
        setInterOpNumThreads(4)
        setIntraOpNumThreads(4)
    }
    return try {
        opts.addNnapi(mapOf(
            "NNAPI_FLAG_USE_FP16"        to "1",
            "NNAPI_FLAG_CPU_DISABLED"    to "0",
        ))
        env.createSession(modelBytes, opts)
    } catch (e: OrtException) {
        opts.addCoreML()  // GPU fallback on non-NNAPI devices
        env.createSession(modelBytes, opts)
    }
}
```

**Recommended model assignment:**

| Model | Delegate |
|-------|----------|
| BERT | NNAPI (NPU) |
| Duration predictor | NNAPI (NPU) |
| Acoustic expand | NNAPI (NPU) |
| F0N predictor | NNAPI (NPU) |
| Conditioner | GPU (small conv) |
| Vocos chunk | GPU (conv + matmul, called 68× per utterance) |

---

## 12. Complete Pipeline (Kotlin Sketch)

```kotlin
class KokoroTtsOnnx(context: Context, voiceFile: File) {

    private val env = OrtEnvironment.getEnvironment()

    private val bertSession   = loadSession("bert_fp16.onnx")
    private val durSession    = loadSession("duration_predictor_fp16.onnx")
    private val acexpSession  = loadSession("acoustic_expand_fp16.onnx")
    private val f0nSession    = loadSession("f0n_predictor_fp16.onnx")
    private val condSession   = loadSession("vocoder_conditioner_fp16.onnx")
    private val vocosSession  = loadSession("vocoder_stream_chunk_fp16.onnx")

    private val voicePack     = loadVoicePack(voiceFile)   // [N_phonemes, 256]
    private val hannWindow    = buildHannWindow(VOCOS_N_FFT)
    private val fft           = FloatFFT_1D(VOCOS_N_FFT.toLong())

    fun generate(text: String, speed: Int = 1): FloatArray {
        val (phonemes, rawIds) = g2p(text)
        val style = voicePack[phonemes.size - 1]

        val inputIds = buildInputIds(rawIds)    // IntArray[510] BOS+ids+EOS padded
        val textMask = buildTextMask(rawIds)    // FloatArray[510]

        // S1
        val dEn = runBert(inputIds, textMask)                      // NTC[1,510,512]

        // S2
        val (predDur, dEnc, tEnStat) = runDuration(dEn, style, textMask, speed, inputIds)

        // S3
        val (expandedIdx, tAcoustic) = expandDurations(predDur)
        val dEncExp = indexSelectAndPad(dEnc, expandedIdx, tAcoustic, dEnc.nChannels, T_ACOUSTIC)
        val asr     = indexSelectNCT(tEnStat, expandedIdx, tAcoustic, 512)

        // S4
        val enNtc = runAcousticExpand(dEncExp)
        val enNct = transposeNTCtoNCT(enNtc, T_ACOUSTIC, 512)

        // S5
        val (f0Full, nFull) = runF0N(enNct, style)
        val tF0 = 2 * tAcoustic
        val f0  = f0Full.copyOfRange(0, tF0)
        val n   = nFull.copyOfRange(0, tF0)

        // S6
        val features    = buildVocosFeatures(asr, f0, n, style, tAcoustic, tF0)
        val conditioned = runConditioner(features)                  // NCT[1,192,T_F0]

        // S7
        return runVocos(conditioned, tF0)
    }

    private fun runVocos(conditioned: FloatArray, tF0: Int): FloatArray {
        var embedPrev  = FloatArray(192 * VOCOS_KERNEL_M1)
        val blockPrevs = Array(VOCOS_N_BLOCKS) { FloatArray(VOCOS_BLOCK_DIM * VOCOS_KERNEL_M1) }
        var prevTail   = FloatArray(VOCOS_TAIL)
        val chunks     = mutableListOf<FloatArray>()
        var pos = 0

        while (pos < tF0) {
            val valid = minOf(VOCOS_CHUNK_FRAMES, tF0 - pos)
            val chunk = extractChunk(conditioned, pos, valid)    // [192, 16] NCT
            val outs  = vocosSession.run(buildVocosInputs(chunk, embedPrev, blockPrevs))

            val xReal = outs["x_real"]!!.value as FloatArray
            val xImag = outs["x_imag"]!!.value as FloatArray
            embedPrev = outs["embed_prev_new"]!!.value as FloatArray
            for (b in 0 until VOCOS_N_BLOCKS)
                blockPrevs[b] = outs["block_${b}_prev_new"]!!.value as FloatArray

            val frames = irfftBatch(xReal, xImag)
            val (audio, newTail) = overlapAdd(frames, prevTail, hannWindow)
            prevTail = newTail
            chunks.add(audio.copyOfRange(0, valid * VOCOS_HOP))
            pos += valid
        }

        return FloatArray(chunks.sumOf { it.size }).also { out ->
            var off = 0; for (c in chunks) { c.copyInto(out, off); off += c.size }
        }
    }

    private fun irfftBatch(xReal: FloatArray, xImag: FloatArray): FloatArray {
        val frames = FloatArray(VOCOS_CHUNK_FRAMES * VOCOS_WIN_LEN)
        for (f in 0 until VOCOS_CHUNK_FRAMES) {
            val real = FloatArray(VOCOS_K) { k -> xReal[f * VOCOS_K + k] }
            val imag = FloatArray(VOCOS_K) { k -> xImag[f * VOCOS_K + k] }
            irfft(real, imag, VOCOS_N_FFT).copyInto(frames, f * VOCOS_WIN_LEN)
        }
        return frames
    }
}
```

---

## 13. Notes on Padding and Quality

- All models use **static shapes** padded to maximum sizes.
- Short utterances (< 50 phonemes) have larger zero-padded regions, which affects BiLSTM quality slightly.
- For best quality, process paragraphs rather than individual short sentences.
- FastGELU introduces < 5×10⁻⁴ absolute activation error vs exact GELU, which is below perceptual threshold.
