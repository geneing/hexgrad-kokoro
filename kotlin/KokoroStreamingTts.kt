package com.kokoro.tts

import android.content.Context
import ai.onnxruntime.*
import java.io.File
import java.util.EnumSet
import kotlin.math.min
import kotlin.math.floor
import kotlin.math.ceil

/**
 * KokoroStreamingTts — Streaming TTS inference for Android using ONNX Runtime.
 *
 * Mirrors the 7-stage Python ONNX pipeline from streaming_vocos_export.ipynb (cell 14):
 *   1. BERT encoder (text → semantic features)
 *   2. Duration predictor (semantic → durations)
 *   3. Duration expansion (Kotlin, index generation)
 *   4. Gather acoustic features (Kotlin, index-based slicing)
 *   5. Acoustic expand (BiLSTM to align with vocoder mesh)
 *   6. F0/N predictor (acoustic → prosody)
 *   7. Vocos feature assembly (Kotlin, channel concatenation & interpolation)
 *   8. Vocos conditioner (features → conditional MelSpectrum)
 *   9. Streaming vocos chunks (stateful backbone decoder with OLA)
 *
 * Audio output: 16-bit PCM @ 24 kHz via callback on each streaming chunk.
 *
 * Usage:
 *   val tts = KokoroStreamingTts(context, ExecutionProvider.CPU, useNnapi = false, useFp16 = true)
 *   tts.loadWeights(File(modelDir))
 *   tts.synthSpeech(inputIds, textMask, style, speed) { shortAudio ->
 *       audioTrack.write(shortAudio, 0, shortAudio.size)
 *   }
 *   tts.close()
 */

// ═══════════════════════════════════════════════════════════════════════════
//  CONSTANTS & DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════

enum class ExecutionProvider {
    CPU,
    NNAPI,
}

data class SemanticResult(
    val conditioned: FloatArray,           // [1, condChannels, 2*T_acoustic]
    val condChannels: Int,                 // typically 640 or similar
    val totalFrames: Int,                  // 2*T_acoustic
    val tAcoustic: Int,                    // actual acoustic frames after duration expand
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SemanticResult) return false
        if (!conditioned.contentEquals(other.conditioned)) return false
        if (condChannels != other.condChannels) return false
        if (totalFrames != other.totalFrames) return false
        if (tAcoustic != other.tAcoustic) return false
        return true
    }

    override fun hashCode(): Int {
        var result = conditioned.contentHashCode()
        result = 31 * result + condChannels
        result = 31 * result + totalFrames
        result = 31 * result + tAcoustic
        return result
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN CLASS
// ═══════════════════════════════════════════════════════════════════════════

class KokoroStreamingTts(
    context: Context,
    private val executionProvider: ExecutionProvider = ExecutionProvider.CPU,
    private val useNnapi: Boolean = false,
    private val nnapiFlags: Int = 0,
    private val cpuThreads: Int = 4,
    private val useFp16: Boolean = true,
) {
    companion object {
        // Kokoro pipeline constants
        const val MAX_INPUT_LENGTH = 510
        const val T_ACOUSTIC_MAX = 8096

        // Vocos streaming constants
        const val VOCOS_CHUNK_FRAMES = 16
        const val HOP_LENGTH = 300
        const val SAMPLES_PER_CHUNK = VOCOS_CHUNK_FRAMES * HOP_LENGTH  // 4800

        // Vocos state dimensions
        const val EMBED_IN = 192
        const val BLOCK_DIM = 384
        const val KERNEL_M1 = 6                              // kernel_size - 1 = 7 - 1
        const val ISTFT_TAIL = 900                           // win_length - hop = 1200 - 300
        const val N_LAYERS = 8                               // ConvNeXt blocks

        const val SAMPLE_RATE = 24000
        const val N_STATE_BUFFERS = 10                       // embed + 8 blocks + istft
    }

    // ONNX Runtime environment & sessions
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessionOptions = SessionOptions().apply {
        intraOpNumThreads = cpuThreads
        interOpNumThreads = cpuThreads
        optimizationLevel = OrtOptimizationLevel.ALL

        if (useNnapi) {
            val flags = if (nnapiFlags != 0) EnumSet.noneOf(NnapiFlags::class.java) else {
                // Default: allow all NNAPI acceleration
                EnumSet.allOf(NnapiFlags::class.java)
            }
            if (nnapiFlags != 0) {
                // User-supplied bitmap of NnapiFlags
                // Note: EnumSet doesn't support direct bitmap construction;
                // caller should pass filter via alternative mechanism if needed
            }
            addNnapi(flags)
        }
    }

    private lateinit var bertSession: OrtSession
    private lateinit var durationSession: OrtSession
    private lateinit var acousticExpandSession: OrtSession
    private lateinit var f0nSession: OrtSession
    private lateinit var conditionerSession: OrtSession
    private lateinit var streamChunkSession: OrtSession

    // ─────────────────────────────────────────────────────────────────────
    //  WEIGHT LOADING
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Load all ONNX model weights from modelDir.
     *
     * Model files loaded depend on the `useFp16` parameter:
     *   - If useFp16=true: bert_fp16.onnx, duration_predictor_fp16.onnx, etc.
     *   - If useFp16=false: bert_opt.onnx, duration_predictor_opt.onnx, etc.
     *
     * @param modelDir Directory containing ONNX model files
     */
    fun loadWeights(modelDir: File) {
        requireNotNull(modelDir) { "modelDir must not be null" }
        require(modelDir.isDirectory) { "modelDir must be a directory" }

        val suffix = if (useFp16) "_fp16.onnx" else "_opt.onnx"

        bertSession = env.createSession(
            readModelBytes(modelDir, "bert$suffix"),
            sessionOptions
        )
        durationSession = env.createSession(
            readModelBytes(modelDir, "duration_predictor$suffix"),
            sessionOptions
        )
        acousticExpandSession = env.createSession(
            readModelBytes(modelDir, "acoustic_expand$suffix"),
            sessionOptions
        )
        f0nSession = env.createSession(
            readModelBytes(modelDir, "f0n_predictor$suffix"),
            sessionOptions
        )
        conditionerSession = env.createSession(
            readModelBytes(modelDir, "vocoder_conditioner$suffix"),
            sessionOptions
        )
        streamChunkSession = env.createSession(
            readModelBytes(modelDir, "vocoder_stream_chunk$suffix"),
            sessionOptions
        )
    }

    private fun readModelBytes(modelDir: File, filename: String): ByteArray {
        val file = File(modelDir, filename)
        require(file.exists()) { "Model file not found: $filename" }
        return file.readBytes()
    }

    // ─────────────────────────────────────────────────────────────────────
    //  SEMANTIC PROCESSING (7-stage pipeline)
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Semantic Processing: Text → conditioned Vocos features
     *
     * Stages:
     *   a. BERT encoder: input_ids, text_mask → d_en
     *   b. Duration predictor: d_en, style, text_mask, speed, input_ids → pred_dur, d_enc, t_en
     *   c. Duration expansion (Kotlin): pred_dur → indices, T_acoustic
     *   d. Gather (Kotlin): index d_enc, t_en with expanded indices
     *   e. Acoustic expand (BiLSTM): d_enc_exp → en
     *   f. F0/N predictor: en, style → F0, N
     *   g. Feature assembly (Kotlin): asr, F0, N, style → features [1, 642, 2*T]
     *   h. Vocos conditioner: features → conditioned
     *
     * @param inputIds     [1, MAX_INPUT_LENGTH] int32 — token IDs (padded)
     * @param textMask     [1, MAX_INPUT_LENGTH] float32 — attention mask (1=token, 0=pad)
     * @param style        [1, 256] float32 — voice style embedding
     * @param speed        [1] int32 — speed multiplier (typically 1)
     * @return SemanticResult containing conditioned features and metadata
     */
    fun semanticProcess(
        inputIds: IntArray,
        textMask: FloatArray,
        style: FloatArray,
        speed: IntArray,
    ): SemanticResult {
        require(inputIds.size == MAX_INPUT_LENGTH) { "inputIds must be $MAX_INPUT_LENGTH long" }
        require(textMask.size == MAX_INPUT_LENGTH) { "textMask must be $MAX_INPUT_LENGTH long" }
        require(style.size == 256) { "style must be 256 long" }
        require(speed.size == 1) { "speed must be length 1" }

        // ── Stage a: BERT Encoder ──
        val dEn = bertSession.run(
            mapOf(
                "input_ids" to createInt32Tensor(env, inputIds, longArrayOf(1, MAX_INPUT_LENGTH.toLong())),
                "text_mask" to createFloat32Tensor(env, textMask, longArrayOf(1, MAX_INPUT_LENGTH.toLong())),
            )
        )
        val dEnOut = (dEn.get(0) as OnnxTensor).floatBuffer.array()  // [1, 512, 510]
        dEn.close()

        // ── Stage b: Duration Predictor ──
        val durResult = durationSession.run(
            mapOf(
                "d_en" to createFloat32Tensor(env, dEnOut, longArrayOf(1, 512, MAX_INPUT_LENGTH.toLong())),
                "style" to createFloat32Tensor(env, style, longArrayOf(1, 256)),
                "text_mask" to createFloat32Tensor(env, textMask, longArrayOf(1, MAX_INPUT_LENGTH.toLong())),
                "speed" to createInt32Tensor(env, speed, longArrayOf(1)),
                "input_ids" to createInt32Tensor(env, inputIds, longArrayOf(1, MAX_INPUT_LENGTH.toLong())),
            )
        )
        val predDur = (durResult.get(0) as OnnxTensor).floatBuffer.array()              // [510]
        val dEnc = (durResult.get(1) as OnnxTensor).floatBuffer.array()                 // [1, h, 510]
        val tEnStatic = (durResult.get(2) as OnnxTensor).floatBuffer.array()            // [1, 512, 510]
        durResult.close()

        // Get h from d_enc shape
        val hDim = dEnc.size / (1 * MAX_INPUT_LENGTH)

        // ── Stage c: Duration Expansion (Kotlin) ──
        val (expandedIndices, tAcoustic) = expandDurations(predDur)

        // ── Stage d: Gather (Kotlin) ──
        val dEncExp = gatherAlongLastAxis(dEnc, hDim, MAX_INPUT_LENGTH, expandedIndices, tAcoustic)  // [1, h, T]
        val asr = gatherAlongLastAxis(tEnStatic, 512, MAX_INPUT_LENGTH, expandedIndices, tAcoustic)  // [1, 512, T]

        // ── Stage e: Acoustic Expand ──
        val acexpResult = acousticExpandSession.run(
            mapOf(
                "d_enc_expanded" to createFloat32Tensor(
                    env, dEncExp, longArrayOf(1, hDim.toLong(), tAcoustic.toLong())
                ),
            )
        )
        val en = (acexpResult.get(0) as OnnxTensor).floatBuffer.array()                 // [1, T, h']
        acexpResult.close()

        val hPrime = en.size / (1 * tAcoustic)

        // ── Stage f: F0/N Predictor ──
        val f0nResult = f0nSession.run(
            mapOf(
                "en" to createFloat32Tensor(env, en, longArrayOf(1, tAcoustic.toLong(), hPrime.toLong())),
                "style" to createFloat32Tensor(env, style, longArrayOf(1, 256)),
            )
        )
        val f0Pred = (f0nResult.get(0) as OnnxTensor).floatBuffer.array()               // [1, 2*T]
        val nPred = (f0nResult.get(1) as OnnxTensor).floatBuffer.array()                // [1, 2*T]
        f0nResult.close()

        val tF0 = f0Pred.size / 1

        // ── Stage g: Feature Assembly (Kotlin) ──
        val features = buildVocosFeatures(asr, tAcoustic, f0Pred, nPred, style, tF0)  // [1, 642, 2*T]

        // ── Stage h: Vocos Conditioner ──
        val condResult = conditionerSession.run(
            mapOf(
                "features" to createFloat32Tensor(env, features, longArrayOf(1, 642, tF0.toLong())),
            )
        )
        val conditioned = (condResult.get(0) as OnnxTensor).floatBuffer.array()
        val condChannels = conditioned.size / (1 * tF0)
        condResult.close()

        return SemanticResult(
            conditioned = conditioned,
            condChannels = condChannels,
            totalFrames = tF0,
            tAcoustic = tAcoustic,
        )
    }

    // ─────────────────────────────────────────────────────────────────────
    //  HELPER FUNCTIONS (Stage c, d, g)
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Stage c: Duration Expansion — Convert per-phoneme durations to frame indices.
     *
     * Algorithm:
     *   1. Cumsum of predDur → boundaries[510]
     *   2. T_acoustic = min(floor(boundaries[509]), T_ACOUSTIC_MAX)
     *   3. For each t in 0..T_acoustic-1, find phoneme index via two-pointer scan
     *
     * @param predDur [510] rounded per-phoneme durations
     * @return (expandedIndices [T_acoustic], T_acoustic)
     */
    private fun expandDurations(predDur: FloatArray): Pair<IntArray, Int> {
        require(predDur.size == MAX_INPUT_LENGTH) { "predDur must be $MAX_INPUT_LENGTH" }

        // Cumsum: boundaries[i] = sum(predDur[0..i])
        val boundaries = FloatArray(MAX_INPUT_LENGTH)
        var cumsum = 0f
        for (i in predDur.indices) {
            cumsum += predDur[i]
            boundaries[i] = cumsum
        }

        // T_acoustic = min(floor(boundaries[509]), T_ACOUSTIC_MAX)
        val tAcoustic = min(boundaries[MAX_INPUT_LENGTH - 1].toInt(), T_ACOUSTIC_MAX)

        // Expand: for each frame t, find phoneme p such that boundaries[p-1] < t <= boundaries[p]
        val expandedIndices = IntArray(tAcoustic)
        var phoneme = 0
        for (t in 0 until tAcoustic) {
            val targetSum = (t + 1).toFloat()  // Frame t corresponds to cumsum interval
            while (phoneme < MAX_INPUT_LENGTH && boundaries[phoneme] < targetSum) {
                phoneme++
            }
            expandedIndices[t] = minOf(phoneme, MAX_INPUT_LENGTH - 1)
        }

        return Pair(expandedIndices, tAcoustic)
    }

    /**
     * Stage d: Gather along last axis — index sequence using expanded indices.
     *
     * Given src [1, channels, srcLen] and indices [tAcoustic], produce
     * dst [1, channels, tAcoustic] where dst[c, t] = src[c, indices[t]].
     *
     * @param src          [1, channels, srcLen] float — C-first layout
     * @param channels     Number of channels
     * @param srcLen       Source length (typically 510)
     * @param indices      [tAcoustic] indices to gather
     * @param tAcoustic    Output sequence length
     * @return [channels * tAcoustic] flat array (C-first: [c, t] → dst[c*T+t])
     */
    private fun gatherAlongLastAxis(
        src: FloatArray,
        channels: Int,
        srcLen: Int,
        indices: IntArray,
        tAcoustic: Int,
    ): FloatArray {
        val dst = FloatArray(channels * tAcoustic)
        for (c in 0 until channels) {
            for (t in 0 until tAcoustic) {
                val idx = indices[t].coerceIn(0, srcLen - 1)
                dst[c * tAcoustic + t] = src[c * srcLen + idx]
            }
        }
        return dst
    }

    /**
     * Stage g: Build Vocos features — assemble [1, 642, T_f0] from components.
     *
     * Feature channels (C-first layout [c, t]):
     *   - [0:512]     ASR features (linearly interpolated from T_acoustic → T_f0)
     *   - [512]       F0 predictor output
     *   - [513]       N (energy) predictor output
     *   - [514:642]   Style embedding (128 channels, broadcasted)
     *
     * Linear interpolation (align_corners=False):
     *   x_src = (i + 0.5) * T_src / T_dst - 0.5
     *   clamp x_src to [0, T_src - 1]
     *   dst[i] = src[floor(x_src)] * (1 - frac) + src[ceil(x_src)] * frac
     *
     * @param asr      [1, 512, T_acoustic] — ASR features (C-first)
     * @param tAsr     T_acoustic (source length)
     * @param f0Pred   [1, T_f0] — F0 predictor output (time-last dense)
     * @param nPred    [1, T_f0] — N predictor output (time-last dense)
     * @param style    [1, 256] — style embedding (use [0:128])
     * @param tF0      T_f0 (target feature length, = 2*T_acoustic)
     * @return [1, 642, T_f0] flat array (C-first)
     */
    private fun buildVocosFeatures(
        asr: FloatArray,
        tAsr: Int,
        f0Pred: FloatArray,
        nPred: FloatArray,
        style: FloatArray,
        tF0: Int,
    ): FloatArray {
        val dst = FloatArray(642 * tF0)

        // ── Interpolate ASR [0:512, :] ──
        for (c in 0 until 512) {
            for (t in 0 until tF0) {
                val xSrc = (t + 0.5f) * tAsr / tF0 - 0.5f
                val xClamped = xSrc.coerceIn(0f, (tAsr - 1).toFloat())
                val lo = xClamped.toInt()
                val hi = minOf(lo + 1, tAsr - 1)
                val frac = xClamped - lo
                val v0 = asr[c * tAsr + lo]
                val v1 = asr[c * tAsr + hi]
                dst[c * tF0 + t] = v0 * (1 - frac) + v1 * frac
            }
        }

        // ── Copy F0 [512, :] ──
        for (t in 0 until tF0) {
            dst[512 * tF0 + t] = f0Pred[t]
        }

        // ── Copy N [513, :] ──
        for (t in 0 until tF0) {
            dst[513 * tF0 + t] = nPred[t]
        }

        // ── Broadcast style [514:642, :] ──
        for (c in 0 until 128) {
            val styleBias = (514 + c)
            for (t in 0 until tF0) {
                dst[styleBias * tF0 + t] = style[c]
            }
        }

        return dst
    }

    // ─────────────────────────────────────────────────────────────────────
    //  STREAMING VOCOS DECODER
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Streaming Vocos Decode — process conditioned features as 16-frame chunks.
     *
     * Manages explicit state buffers (embed, 8 blocks, ISTFT) and yields audio
     * on each streaming chunk. Called by synthSpeech().
     *
     * @param result           SemanticResult from semanticProcess()
     * @param onChunk          Callback: (FloatArray) → Boolean. Return false to stop.
     */
    private fun streamingVocosDecode(
        result: SemanticResult,
        onChunk: (FloatArray) -> Boolean,
    ) {
        val conditioned = result.conditioned
        val condChannels = result.condChannels
        val totalFrames = result.totalFrames

        // Initialize state buffers (10 total)
        val embedPrev = FloatArray(EMBED_IN * KERNEL_M1)       // [1, 192, 6]
        val blockPrevs = Array(N_LAYERS) { FloatArray(BLOCK_DIM * KERNEL_M1) }  // 8x [1, 384, 6]
        val istftPrev = FloatArray(ISTFT_TAIL)                // [1, 900]

        var pos = 0
        while (pos < totalFrames) {
            val end = minOf(totalFrames, pos + VOCOS_CHUNK_FRAMES)
            val valid = end - pos

            // Slice chunk from conditioned and pad to VOCOS_CHUNK_FRAMES
            val chunkFlat = FloatArray(condChannels * VOCOS_CHUNK_FRAMES)
            for (c in 0 until condChannels) {
                for (f in 0 until VOCOS_CHUNK_FRAMES) {
                    val srcIdx = if (pos + f < totalFrames) pos + f else totalFrames - 1
                    chunkFlat[c * VOCOS_CHUNK_FRAMES + f] = conditioned[c * totalFrames + srcIdx]
                }
            }

            // Build input map for streamChunkSession
            val inputs = mutableMapOf<String, OnnxTensor>(
                "conditioned_chunk" to createFloat32Tensor(
                    env, chunkFlat, longArrayOf(1, condChannels.toLong(), VOCOS_CHUNK_FRAMES.toLong())
                ),
                "embed_prev" to createFloat32Tensor(env, embedPrev, longArrayOf(1, EMBED_IN.toLong(), KERNEL_M1.toLong())),
            )
            for (i in 0 until N_LAYERS) {
                inputs["block_${i}_prev"] = createFloat32Tensor(
                    env, blockPrevs[i], longArrayOf(1, BLOCK_DIM.toLong(), KERNEL_M1.toLong())
                )
            }
            inputs["istft_prev"] = createFloat32Tensor(env, istftPrev, longArrayOf(1, ISTFT_TAIL.toLong()))

            // Run streaming chunk
            val outs = streamChunkSession.run(inputs)

            // Extract audio output and trim to valid samples
            val audioFull = (outs.get(0) as OnnxTensor).floatBuffer.array()  // [1, 4800]
            val audioTrimmed = FloatArray(valid * HOP_LENGTH)
            for (i in 0 until valid * HOP_LENGTH) {
                audioTrimmed[i] = audioFull[i]
            }

            // Update state buffers
            val embedPrevNew = (outs.get(1) as OnnxTensor).floatBuffer.array()
            embedPrev.indices.forEach { i -> embedPrev[i] = embedPrevNew[i] }

            for (i in 0 until N_LAYERS) {
                val blockNew = (outs.get(2 + i) as OnnxTensor).floatBuffer.array()
                blockPrevs[i].indices.forEach { j -> blockPrevs[i][j] = blockNew[j] }
            }

            val istftPrevNew = (outs.get(2 + N_LAYERS) as OnnxTensor).floatBuffer.array()
            istftPrev.indices.forEach { i -> istftPrev[i] = istftPrevNew[i] }

            // Close all tensors
            inputs.values.forEach { it.close() }
            outs.close()

            // Invoke callback
            if (!onChunk(audioTrimmed)) {
                break
            }

            pos = end
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  TTS SPEECH LOOP
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Synthesize speech — full TTS pipeline with streaming audio callback.
     *
     * Steps:
     *   1. semanticProcess: text → conditioned Vocos features
     *   2. streamingVocosDecode: features → audio chunks
     *   3. Convert float audio to 16-bit PCM and deliver via callback
     *
     * @param inputIds    [1, MAX_INPUT_LENGTH] int32 — token IDs (padded)
     * @param textMask    [1, MAX_INPUT_LENGTH] float32 — attention mask
     * @param style       [1, 256] float32 — voice style embedding
     * @param speed       [1] int32 — speed multiplier (typically 1)
     * @param onChunk     Callback: (ShortArray) → Boolean. Receives 16-bit PCM audio chunk.
     *                    Return false to stop synthesis.
     */
    fun synthSpeech(
        inputIds: IntArray,
        textMask: FloatArray,
        style: FloatArray,
        speed: IntArray,
        onChunk: (ShortArray) -> Boolean,
    ) {
        // Stage 1–8: Semantic processing
        val result = semanticProcess(inputIds, textMask, style, speed)

        // Stage 9: Streaming vocos decode with audio conversion
        streamingVocosDecode(result) { floatAudio ->
            // Convert float [-1, 1] to int16 PCM
            val shortAudio = ShortArray(floatAudio.size)
            for (i in floatAudio.indices) {
                val sample = floatAudio[i].coerceIn(-1f, 1f)
                shortAudio[i] = (sample * 32767).toInt().toShort()
            }
            onChunk(shortAudio)
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  CLEANUP
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Close all ONNX Runtime resources.
     * Call when done with inference.
     */
    fun close() {
        if (::bertSession.isInitialized) bertSession.close()
        if (::durationSession.isInitialized) durationSession.close()
        if (::acousticExpandSession.isInitialized) acousticExpandSession.close()
        if (::f0nSession.isInitialized) f0nSession.close()
        if (::conditionerSession.isInitialized) conditionerSession.close()
        if (::streamChunkSession.isInitialized) streamChunkSession.close()
        sessionOptions.close()
        env.close()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  TENSOR FACTORY HELPERS
// ═══════════════════════════════════════════════════════════════════════════

fun createFloat32Tensor(
    env: OrtEnvironment,
    data: FloatArray,
    shape: LongArray,
): OnnxTensor {
    val buffer = java.nio.FloatBuffer.wrap(data)
    return OnnxTensor.createTensor(env, buffer, shape)
}

fun createInt32Tensor(
    env: OrtEnvironment,
    data: IntArray,
    shape: LongArray,
): OnnxTensor {
    val buffer = java.nio.IntBuffer.wrap(data)
    return OnnxTensor.createTensor(env, buffer, shape)
}
