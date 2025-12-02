package com.yjcho.cifar10tflite

import android.content.Context
import android.util.Log
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.DownloadType
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.system.measureNanoTime

class TFLiteClassifier(
    private val context: Context,
    private val modelPath: String = "base_int8.tflite",
    private val useGpu: Boolean = false,
    private val numThreads: Int = 4,
    private val firebaseModelName: String = "cifar10_int8"  // Firebase 모델 이름
) {
    private var interpreter: Interpreter? = null

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
        loadFirebaseModel()
    }

    /**
     * Firebase에서 모델 로드 (Fallback: 로컬 모델)
     */
    private fun loadFirebaseModel() {
        Log.d(TAG, "Firebase 모델 로드 시도: $firebaseModelName")

        val conditions = CustomModelDownloadConditions.Builder()
            .requireWifi()
            .build()

        FirebaseModelDownloader.getInstance()
            .getModel(firebaseModelName, DownloadType.LOCAL_MODEL, conditions)
            .addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    val modelFile: File? = task.result.file

                    if (modelFile != null && modelFile.exists()) {
                        try {
                            interpreter = Interpreter(loadModelFromFile(modelFile))
                            Log.d(TAG, "Firebase 모델 로드 완료: ${modelFile.name}")
                            logModelInfo()
                        } catch (e: Exception) {
                            Log.e(TAG, "Firebase 모델 로드 실패, 로컬 모델 사용", e)
                            loadLocalModel()
                        }
                    } else {
                        Log.w(TAG, "Firebase 모델 파일 없음, 로컬 모델 사용")
                        loadLocalModel()
                    }
                } else {
                    Log.e(TAG, "Firebase 모델 다운로드 실패, 로컬 모델 사용", task.exception)
                    loadLocalModel()
                }
            }
    }

    /**
     * 로컬(assets) 모델 로드
     */
    private fun loadLocalModel() {
        try {
            Log.d(TAG, "로컬 모델 로드 시도: $modelPath")

            val fd = context.assets.openFd(modelPath)
            val inputStream = FileInputStream(fd.fileDescriptor)
            val fileChannel = inputStream.channel
            val buffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )

            interpreter = Interpreter(buffer)
            Log.d(TAG, "로컬 모델 로드 완료")
            logModelInfo()
        } catch (e: Exception) {
            Log.e(TAG, "로컬 모델 로드 실패", e)
            throw e
        }
    }

    /**
     * Firebase 모델 파일을 MappedByteBuffer로 로드
     */
    private fun loadModelFromFile(file: File): MappedByteBuffer {
        val inputStream = FileInputStream(file)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            0,
            file.length()
        )
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
        }
    }

    fun classify(imageBytes: FloatArray): ClassificationResult {
        require(imageBytes.size == 32 * 32 * 3) {
            "이미지 크기가 맞지 않습니다: ${imageBytes.size}"
        }

        var resultClassName = ""
        var resultConfidence = 0f
        val resultProbabilities = FloatArray(10)

        val inferenceTime = measureNanoTime {
            // 1. 입력 버퍼 생성 (UINT8)
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

            // 4. 추론 실행
            interpreter?.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()

            // 5. 출력 해석: UINT8 -> 확률
            val rawOutput = ByteArray(10)
            outputBuffer.get(rawOutput)

            val logits = FloatArray(10)
            for (i in 0 until 10) {
                val value = rawOutput[i].toInt() and 0xFF
                logits[i] = value.toFloat() / 255f
            }

            // 6. Softmax 적용
            val maxLogit = logits.maxOrNull() ?: 0f
            var sum = 0f
            for (i in 0 until 10) {
                logits[i] = exp(logits[i] - maxLogit)
                sum += logits[i]
            }
            for (i in 0 until 10) {
                logits[i] /= sum
                resultProbabilities[i] = logits[i]
            }

            // 7. 최대 확률 찾기
            val maxIndex = logits.indices.maxByOrNull { logits[it] } ?: 0
            resultClassName = CIFAR10_CLASSES[maxIndex]
            resultConfidence = logits[maxIndex]
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
    }
}