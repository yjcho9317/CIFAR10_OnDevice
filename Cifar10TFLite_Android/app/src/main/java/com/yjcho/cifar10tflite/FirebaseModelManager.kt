package com.yjcho.cifar10tflite

import android.content.Context
import android.util.Log
import com.google.firebase.ml.modeldownloader.CustomModel
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.DownloadType
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class FirebaseModelManager(private val context: Context) {
    
    private var interpreter: Interpreter? = null
    private var isRemoteModel = false
    
    companion object {
        private const val TAG = "FirebaseModelManager"
    }
    
    /**
     * Firebase ML 모델 초기화
     * 
     * @param localModelPath assets의 내장 모델 경로
     * @param remoteModelName Firebase Console에 업로드한 모델 이름
     * @param onReady 모델 준비 완료 콜백 (isRemote: Boolean)
     * @param onError 오류 발생 콜백
     */
    fun initialize(
        localModelPath: String = "base_int8.tflite",
        remoteModelName: String = "cifar10_model",
        onReady: (Boolean) -> Unit,
        onError: (Exception) -> Unit
    ) {
        Log.d(TAG, "모델 초기화 시작...")
        
        // WiFi에서만 다운로드
        val conditions = CustomModelDownloadConditions.Builder()
            .requireWifi()
            .build()
        
        FirebaseModelDownloader.getInstance()
            .getModel(remoteModelName, DownloadType.LOCAL_MODEL, conditions)
            .addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    val model: CustomModel = task.result
                    val modelFile: File? = model.file
                    
                    if (modelFile != null && modelFile.exists()) {
                        try {
                            // Firebase 모델 로드
                            interpreter = Interpreter(loadModelFromFile(modelFile))
                            isRemoteModel = true
                            
                            Log.d(TAG, "Firebase 모델 로드 완료: ${modelFile.name}")
                            Log.d(TAG, "모델 크기: ${modelFile.length() / 1024}KB")
                            onReady(true)
                        } catch (e: Exception) {
                            Log.e(TAG, "Firebase 모델 로드 실패", e)
                            loadLocalModel(localModelPath, onReady, onError)
                        }
                    } else {
                        Log.w(TAG, "Firebase 모델 파일 없음, 내장 모델 사용")
                        loadLocalModel(localModelPath, onReady, onError)
                    }
                } else {
                    // 다운로드 실패 (네트워크 오류 등)
                    Log.e(TAG, "Firebase 모델 다운로드 실패", task.exception)
                    loadLocalModel(localModelPath, onReady, onError)
                }
            }
    }
    
    /**
     * Firebase에서 다운로드한 모델 파일을 MappedByteBuffer로 로드
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
     * assets의 내장 모델 로드 (Fallback)
     */
    private fun loadLocalModel(
        modelPath: String,
        onReady: (Boolean) -> Unit,
        onError: (Exception) -> Unit
    ) {
        try {
            Log.d(TAG, "내장 모델 로드 시도: $modelPath")
            
            val fd = context.assets.openFd(modelPath)
            val inputStream = FileInputStream(fd.fileDescriptor)
            val fileChannel = inputStream.channel
            val buffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )
            
            interpreter = Interpreter(buffer)
            isRemoteModel = false
            
            Log.d(TAG, "내장 모델 로드 완료")
            onReady(false)
        } catch (e: Exception) {
            Log.e(TAG, "내장 모델 로드 실패", e)
            onError(e)
        }
    }
    
    fun getInterpreter(): Interpreter? = interpreter
    
    fun isUsingRemoteModel(): Boolean = isRemoteModel
    
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}