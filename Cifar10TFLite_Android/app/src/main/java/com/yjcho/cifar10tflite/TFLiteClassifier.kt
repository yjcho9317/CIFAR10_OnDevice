package com.yjcho.cifar10tflite

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.system.measureNanoTime

class TFLiteClassifier(
    private val context: Context,
    private val modelPath: String = "base_int8.tflite",
    private val useGpu: Boolean = false,
    private val numThreads: Int = 4
) {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    companion object {
        private const val TAG = "TFLiteClassifier"

        val CIFAR10_CLASSES = arrayOf(
            "비행기", "자동차", "새", "고양이", "사슴",
            "개", "개구리", "말", "배", "트럭"
        )
    }

    data class ClassificationResult(
        val className: String,
        val confidence: Float,
        val inferenceTimeMs: Double,
        val allProbabilities: FloatArray
    )

    init {
        loadLocalModel()
    }

    /**
     * Interpreter.Options 생성
     * - numThreads 적용
     * - useGpu=true면 GpuDelegate 추가 (INT8 지원 옵션 포함)
     */
    private fun buildInterpreterOptions(): Pair<Interpreter.Options, GpuDelegate?> {
        val options = Interpreter.Options()
        options.setNumThreads(numThreads)

        var delegate: GpuDelegate? = null
        if (useGpu) {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                try {
                    // INT8 양자화 모델도 GPU에서 돌릴 수 있게 허용
                    val delegateOptions = GpuDelegate.Options().apply {
                        setQuantizedModelsAllowed(true)
                    }
                    delegate = GpuDelegate(delegateOptions)
                    options.addDelegate(delegate)
                    Log.d(TAG, "GPU Delegate 활성화 (quantized models allowed)")
                } catch (e: Exception) {
                    Log.e(TAG, "GPU Delegate 생성 실패, CPU로 fallback", e)
                    delegate = null
                }
            } else {
                Log.w(TAG, "이 기기는 GPU Delegate 미지원 - CPU로 실행")
            }
        }

        Log.d(TAG, "Interpreter Options: threads=$numThreads, useGpu=$useGpu, gpuActive=${delegate != null}")
        return Pair(options, delegate)
    }

    /**
     * 로컬(assets) 모델 로드 - 동기
     */
    private fun loadLocalModel() {
        try {
            Log.d(TAG, "로컬 모델 로드: $modelPath")

            val fd = context.assets.openFd(modelPath)
            val inputStream = FileInputStream(fd.fileDescriptor)
            val fileChannel = inputStream.channel
            val buffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )

            val (options, delegate) = buildInterpreterOptions()
            interpreter = Interpreter(buffer, options)
            gpuDelegate = delegate
            Log.d(TAG, "로컬 모델 로드 완료")
            logModelInfo()
        } catch (e: Exception) {
            Log.e(TAG, "로컬 모델 로드 실패", e)
            throw e
        }
    }

    /**
     * 모델 정보 로그
     */
    private fun logModelInfo() {
        interpreter?.let {
            val inputTensor = it.getInputTensor(0)
            val outputTensor = it.getOutputTensor(0)

            Log.d(TAG, "Input shape: ${inputTensor.shape().contentToString()}")
            Log.d(TAG, "Output shape: ${outputTensor.shape().contentToString()}")
            Log.d(TAG, "Input type: ${inputTensor.dataType()}")
            Log.d(TAG, "Output type: ${outputTensor.dataType()}")

            val outParams = outputTensor.quantizationParams()
            Log.d(TAG, "Output quant: scale=${outParams.scale}, zeroPoint=${outParams.zeroPoint}")
        }
    }

    fun classify(imageBytes: FloatArray): ClassificationResult {
        require(imageBytes.size == 32 * 32 * 3) {
            "이미지 크기가 맞지 않습니다: ${imageBytes.size}"
        }
        val interp = interpreter ?: error("Interpreter가 초기화되지 않았습니다")

        var resultClassName = ""
        var resultConfidence = 0f
        val resultProbabilities = FloatArray(10)

        val inferenceTime = measureNanoTime {
            // 1. 입력 버퍼 (UINT8)
            val inputBuffer = ByteBuffer.allocateDirect(1 * 32 * 32 * 3).apply {
                order(ByteOrder.nativeOrder())
            }

            // 2. Float (0~1) -> UINT8 (0~255) 변환
            for (i in imageBytes.indices) {
                val pixel = imageBytes[i]
                val uint8Value = (pixel * 255f).toInt().coerceIn(0, 255)
                inputBuffer.put(uint8Value.toByte())
            }
            inputBuffer.rewind()

            // 3. 출력 버퍼 (UINT8)
            val outputBuffer = ByteBuffer.allocateDirect(1 * 10).apply {
                order(ByteOrder.nativeOrder())
            }

            // 4. 추론
            interp.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()

            // 5. 출력 역양자화: UINT8 -> 확률
            //    모델 마지막 레이어가 softmax이므로 이미 확률이다.
            //    scale, zero_point로 역양자화만 하면 끝. softmax 추가 적용 불필요.
            val rawOutput = ByteArray(10)
            outputBuffer.get(rawOutput)

            val outParams = interp.getOutputTensor(0).quantizationParams()
            val scale = outParams.scale
            val zeroPoint = outParams.zeroPoint

            for (i in 0 until 10) {
                val uint8 = rawOutput[i].toInt() and 0xFF
                resultProbabilities[i] = (uint8 - zeroPoint) * scale
            }

            // 6. 최대 확률 클래스
            val maxIndex = resultProbabilities.indices
                .maxByOrNull { resultProbabilities[it] } ?: 0
            resultClassName = CIFAR10_CLASSES[maxIndex]
            resultConfidence = resultProbabilities[maxIndex]
        }

        val inferenceTimeMs = inferenceTime / 1_000_000.0

        return ClassificationResult(
            className = resultClassName,
            confidence = resultConfidence,
            inferenceTimeMs = inferenceTimeMs,
            allProbabilities = resultProbabilities
        )
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
    }
}
