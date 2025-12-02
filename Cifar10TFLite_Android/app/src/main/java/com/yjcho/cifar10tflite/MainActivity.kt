package com.yjcho.cifar10tflite

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext  // 추가
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel  // 이렇게 명시적으로 import
import com.yjcho.cifar10tflite.TestDataLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Cifar10Screen()
                }
            }
        }
    }
}

class Cifar10ViewModel : ViewModel() {
    var uiState by mutableStateOf(Cifar10UiState())
        private set

    data class Cifar10UiState(
        val isLoading: Boolean = false,
        val progress: Int = 0,
        val totalImages: Int = 100,
        val cpuAccuracy: Float? = null,
        val gpuAccuracy: Float? = null,
        val cpuAvgTime: Double? = null,
        val gpuAvgTime: Double? = null,
        val errorMessage: String? = null,
        val testResults: List<TestResult> = emptyList()
    )

    data class TestResult(
        val imageIndex: Int,
        val actualLabel: String,
        val cpuPrediction: String,
        val gpuPrediction: String,
        val cpuCorrect: Boolean,
        val gpuCorrect: Boolean,
        val cpuTime: Double,
        val gpuTime: Double
    )

    fun runTest(context: android.content.Context) {
        viewModelScope.launch {
            uiState = uiState.copy(
                isLoading = true,
                progress = 0,
                errorMessage = null,
                testResults = emptyList()
            )

            try {
                withContext(Dispatchers.IO) {
                    val dataLoader = TestDataLoader(context)

                    // 전체 테스트 개수 설정 (1000개 또는 10000개)
                    val totalTestCount = 1000  // 메모리 절약: 1000개만
                    // val totalTestCount = 10000  // 전체 테스트

                    uiState = uiState.copy(totalImages = totalTestCount)

                    // CPU 테스트
                    val cpuClassifier = TFLiteClassifier(
                        context = context,
                        useGpu = false,
                        numThreads = 4
                    )

                    val cpuResult = dataLoader.calculateAccuracyInBatches(
                        assetPath = "cifar10_test_1000.bin",
                        totalCount = totalTestCount,
                        batchSize = 100,  // 배치 크기
                        classifier = cpuClassifier
                    ) { current, total ->
                        uiState = uiState.copy(progress = current)
                    }

                    cpuClassifier.close()

                    // GPU 테스트
                    val gpuClassifier = TFLiteClassifier(
                        context = context,
                        useGpu = true,
                        numThreads = 4
                    )

                    val gpuResult = dataLoader.calculateAccuracyInBatches(
                        assetPath = "cifar10_test_1000.bin",
                        totalCount = totalTestCount,
                        batchSize = 100,
                        classifier = gpuClassifier
                    ) { current, total ->
                        uiState = uiState.copy(progress = current + totalTestCount)
                    }

                    gpuClassifier.close()

                    // 개별 결과 (처음 20개만)
                    val testResults = mutableListOf<TestResult>()
                    val cpuClassifier2 = TFLiteClassifier(context, useGpu = false)
                    val gpuClassifier2 = TFLiteClassifier(context, useGpu = true)

                    val sampleImages = dataLoader.loadTestData(
                        assetPath = "cifar10_test_1000.bin",
                        startIndex = 0,
                        count = 20
                    )

                    sampleImages.forEachIndexed { index, testImage ->
                        val cpuPred = cpuClassifier2.classify(testImage.pixels)
                        val gpuPred = gpuClassifier2.classify(testImage.pixels)

                        testResults.add(
                            TestResult(
                                imageIndex = index,
                                actualLabel = TFLiteClassifier.CIFAR10_CLASSES[testImage.label],
                                cpuPrediction = cpuPred.className,
                                gpuPrediction = gpuPred.className,
                                cpuCorrect = cpuPred.className == TFLiteClassifier.CIFAR10_CLASSES[testImage.label],
                                gpuCorrect = gpuPred.className == TFLiteClassifier.CIFAR10_CLASSES[testImage.label],
                                cpuTime = cpuPred.inferenceTimeMs,  // 이미 Double
                                gpuTime = gpuPred.inferenceTimeMs   // 이미 Double
                            )
                        )
                    }

                    cpuClassifier2.close()
                    gpuClassifier2.close()

                    withContext(Dispatchers.Main) {
                        uiState = uiState.copy(
                            isLoading = false,
                            cpuAccuracy = cpuResult.accuracy,
                            gpuAccuracy = gpuResult.accuracy,
                            cpuAvgTime = cpuResult.avgInferenceTimeMs,
                            gpuAvgTime = gpuResult.avgInferenceTimeMs,
                            testResults = testResults
                        )
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    uiState = uiState.copy(
                        isLoading = false,
                        errorMessage = "오류 발생: ${e.message}\n${e.stackTraceToString()}"
                    )
                }
            }
        }
    }
}

@Composable
fun Cifar10Screen(viewModel: Cifar10ViewModel = viewModel()) {
    val uiState = viewModel.uiState
    val context = LocalContext.current  // 수정됨

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // 제목
        Text(
            text = "CIFAR-10 TFLite 정확도 테스트",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        // 테스트 시작 버튼
        Button(
            onClick = { viewModel.runTest(context) },
            enabled = !uiState.isLoading,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (uiState.isLoading) "테스트 중..." else "테스트 시작")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 진행상황
        if (uiState.isLoading) {
            LinearProgressIndicator(
                progress = { uiState.progress.toFloat() / (uiState.totalImages * 2) },
                modifier = Modifier.fillMaxWidth()
            )
            Text(
                text = "진행: ${uiState.progress} / ${uiState.totalImages * 2}",
                modifier = Modifier.padding(top = 8.dp)
            )
        }

        // 결과 표시
        if (uiState.cpuAccuracy != null && uiState.gpuAccuracy != null) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "📊 테스트 결과",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // CPU 결과
                    ResultRow(
                        label = "CPU",
                        accuracy = uiState.cpuAccuracy,
                        avgTime = uiState.cpuAvgTime!!,
                        color = Color(0xFF2196F3)
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // GPU 결과
                    ResultRow(
                        label = "GPU",
                        accuracy = uiState.gpuAccuracy,
                        avgTime = uiState.gpuAvgTime!!,
                        color = Color(0xFF4CAF50)
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // 속도 향상
                    val speedup = uiState.cpuAvgTime / uiState.gpuAvgTime
                    Text(
                        text = "⚡ 속도 향상: ${String.format("%.2f", speedup)}배",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFFF9800)
                    )
                }
            }

            // 개별 테스트 결과
            Text(
                text = "🔍 개별 테스트 결과 (상위 20개)",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(vertical = 8.dp)
            )

            LazyColumn {
                items(uiState.testResults) { result ->
                    TestResultCard(result)
                }
            }
        }

        // 에러 메시지
        uiState.errorMessage?.let { error ->
            Text(
                text = error,
                color = Color.Red,
                modifier = Modifier.padding(top = 8.dp)
            )
        }
    }
}

@Composable
fun ResultRow(
    label: String,
    accuracy: Float,
    avgTime: Double,
    color: Color
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column {
            Text(
                text = label,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = color
            )
            Text(
                text = "정확도: ${String.format("%.2f", accuracy * 100)}%",
                fontSize = 14.sp
            )
        }
        Column(horizontalAlignment = Alignment.End) {
            Text(
                text = "${String.format("%.2f", avgTime)}ms",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = color
            )
            Text(
                text = "평균 추론시간",
                fontSize = 12.sp
            )
        }
    }
}

@Composable
fun TestResultCard(result: Cifar10ViewModel.TestResult) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                text = "이미지 #${result.imageIndex} - 정답: ${result.actualLabel}",
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "CPU: ${result.cpuPrediction}",
                        color = if (result.cpuCorrect) Color(0xFF4CAF50) else Color(0xFFF44336),
                        fontSize = 12.sp
                    )
                    Text(
                        text = "${String.format("%.2f", result.cpuTime)}ms",
                        fontSize = 10.sp,
                        color = Color.Gray
                    )
                }

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "GPU: ${result.gpuPrediction}",
                        color = if (result.gpuCorrect) Color(0xFF4CAF50) else Color(0xFFF44336),
                        fontSize = 12.sp
                    )
                    Text(
                        text = "${String.format("%.2f", result.gpuTime)}ms",
                        fontSize = 10.sp,
                        color = Color.Gray
                    )
                }
            }
        }
    }
}