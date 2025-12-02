package com.yjcho.cifar10tflite

import android.content.Context
import android.util.Log
import java.io.BufferedInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TestDataLoader(private val context: Context) {

    companion object {
        private const val TAG = "TestDataLoader"
        private const val IMAGE_SIZE_BYTES = 1 + 3072 * 4  // 1 byte label + 3072 floats
    }

    data class TestImage(
        val pixels: FloatArray,
        val label: Int
    )

    /**
     * 테스트 데이터 로드 (배치 단위)
     * @param assetPath 파일 경로
     * @param startIndex 시작 인덱스 (0부터)
     * @param count 로드할 개수
     */
    fun loadTestData(
        assetPath: String = "cifar10_test_1000.bin",
        startIndex: Int = 0,
        count: Int = 1000
    ): List<TestImage> {
        val testImages = mutableListOf<TestImage>()

        try {
            context.assets.open(assetPath).use { rawStream ->
                val inputStream = BufferedInputStream(rawStream, 8192)

                // startIndex까지 스킵
                val skipBytes = startIndex.toLong() * IMAGE_SIZE_BYTES
                if (skipBytes > 0) {
                    var remaining = skipBytes
                    while (remaining > 0) {
                        val skipped = inputStream.skip(remaining)
                        if (skipped <= 0) break
                        remaining -= skipped
                    }
                    Log.d(TAG, "Skipped to index $startIndex")
                }

                val buffer = ByteArray(IMAGE_SIZE_BYTES)

                repeat(count) { index ->
                    try {
                        val bytesRead = inputStream.read(buffer)
                        if (bytesRead != buffer.size) {
                            Log.w(TAG, "End of file at index ${startIndex + index}")
                            return@repeat
                        }

                        // Label
                        val label = buffer[0].toInt() and 0xFF

                        // Pixels
                        val pixels = FloatArray(32 * 32 * 3)
                        val byteBuffer = ByteBuffer.wrap(buffer, 1, 3072 * 4).apply {
                            order(ByteOrder.LITTLE_ENDIAN)
                        }

                        for (i in pixels.indices) {
                            pixels[i] = byteBuffer.float
                        }

                        testImages.add(TestImage(pixels, label))

                        // 처음 3개와 마지막 로그
                        if (index < 3 || index == count - 1) {
                            Log.d(TAG, "Image ${startIndex + index} - Label: $label")
                        }

                    } catch (e: Exception) {
                        Log.e(TAG, "이미지 ${startIndex + index} 로드 실패", e)
                        return@repeat
                    }
                }
            }

            Log.d(TAG, "테스트 데이터 로드 완료: ${testImages.size}개 (${startIndex}~${startIndex + testImages.size - 1})")

        } catch (e: Exception) {
            Log.e(TAG, "테스트 데이터 로드 실패", e)
        }

        return testImages
    }

    /**
     * 배치 단위로 정확도 계산
     */
    fun calculateAccuracyInBatches(
        assetPath: String = "cifar10_test_1000.bin",
        totalCount: Int = 10000,
        batchSize: Int = 1000,
        classifier: TFLiteClassifier,
        progressCallback: (Int, Int) -> Unit = { _, _ -> }
    ): AccuracyResult {
        var correctCount = 0
        var processedCount = 0
        val confusionMatrix = Array(10) { IntArray(10) }
        val inferenceTimesMs = mutableListOf<Double>()

        val totalBatches = (totalCount + batchSize - 1) / batchSize

        for (batchIndex in 0 until totalBatches) {
            val startIndex = batchIndex * batchSize
            val currentBatchSize = minOf(batchSize, totalCount - startIndex)

            Log.d(TAG, "배치 ${batchIndex + 1}/$totalBatches 처리 중... (${startIndex}~${startIndex + currentBatchSize - 1})")

            // 배치 로드
            val batchImages = loadTestData(assetPath, startIndex, currentBatchSize)

            // 배치 처리
            batchImages.forEach { testImage ->
                val result = classifier.classify(testImage.pixels)

                val predictedClass = TFLiteClassifier.CIFAR10_CLASSES.indexOf(result.className)
                val actualClass = testImage.label

                if (predictedClass == actualClass) {
                    correctCount++
                }

                confusionMatrix[actualClass][predictedClass]++
                inferenceTimesMs.add(result.inferenceTimeMs)

                processedCount++
                progressCallback(processedCount, totalCount)
            }

            // 메모리 정리
            System.gc()
        }

        val accuracy = correctCount.toFloat() / totalCount.toFloat()
        val avgInferenceTime = inferenceTimesMs.average()

        return AccuracyResult(
            accuracy = accuracy,
            correctCount = correctCount,
            totalCount = totalCount,
            confusionMatrix = confusionMatrix,
            avgInferenceTimeMs = avgInferenceTime,
            inferenceTimesMs = inferenceTimesMs
        )
    }

    data class AccuracyResult(
        val accuracy: Float,
        val correctCount: Int,
        val totalCount: Int,
        val confusionMatrix: Array<IntArray>,
        val avgInferenceTimeMs: Double,
        val inferenceTimesMs: List<Double>
    )
}