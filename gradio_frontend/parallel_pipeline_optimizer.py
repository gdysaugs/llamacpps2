"""
並列処理パイプライン最適化モジュール
各処理を並列化し、GPUメモリを効率的に管理
"""
import torch
import gc
import asyncio
import concurrent.futures
from typing import Optional, Tuple, Dict, Any
import threading
import queue
import time
from pathlib import Path
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelPipelineOptimizer:
    """並列処理パイプライン最適化クラス"""

    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.preprocessing_queue = queue.Queue()
        self.results_cache = {}

    def clear_gpu_memory(self):
        """GPUメモリを解放"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            logger.info(f"🧹 GPU Memory cleared. Available: {torch.cuda.mem_get_info()[0] / 1024**3:.2f}GB")

    def monitor_gpu_memory(self):
        """GPUメモリ使用量を監視"""
        if torch.cuda.is_available():
            total = torch.cuda.mem_get_info()[1] / 1024**3
            used = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3
            logger.info(f"📊 GPU Memory: {used:.2f}GB / {total:.2f}GB ({used/total*100:.1f}%)")

    async def parallel_llama_sovits(self,
                                  llama_func,
                                  sovits_prep_func,
                                  prompt: str,
                                  voice_config: Dict) -> Tuple[str, Any]:
        """
        Llama生成とSoVITS前処理を並列実行
        """
        logger.info("🚀 Starting parallel Llama + SoVITS preprocessing")

        # 並列タスク作成
        loop = asyncio.get_event_loop()

        # Llama生成タスク
        llama_future = loop.run_in_executor(
            self.executor,
            llama_func,
            prompt
        )

        # SoVITS前処理タスク
        sovits_prep_future = loop.run_in_executor(
            self.executor,
            sovits_prep_func,
            voice_config
        )

        # 両タスクの完了を待つ
        llama_result, sovits_prep = await asyncio.gather(
            llama_future,
            sovits_prep_future
        )

        # GPU メモリクリア
        self.clear_gpu_memory()

        return llama_result, sovits_prep

    async def parallel_sovits_wav2lip_prep(self,
                                          sovits_func,
                                          wav2lip_prep_func,
                                          text: str,
                                          sovits_config: Any,
                                          video_path: str) -> Tuple[str, Any]:
        """
        SoVITS音声生成とWav2Lip前処理を並列実行
        """
        logger.info("🎵 Starting parallel SoVITS + Wav2Lip preprocessing")

        loop = asyncio.get_event_loop()

        # SoVITS音声生成タスク
        sovits_future = loop.run_in_executor(
            self.executor,
            sovits_func,
            text,
            sovits_config
        )

        # Wav2Lip動画前処理タスク
        wav2lip_prep_future = loop.run_in_executor(
            self.executor,
            wav2lip_prep_func,
            video_path
        )

        # 両タスクの完了を待つ
        audio_path, video_prep = await asyncio.gather(
            sovits_future,
            wav2lip_prep_future
        )

        # GPU メモリクリア
        self.clear_gpu_memory()

        return audio_path, video_prep

    async def parallel_wav2lip_facefusion_prep(self,
                                              wav2lip_func,
                                              facefusion_prep_func,
                                              audio_path: str,
                                              video_data: Any,
                                              source_face: str) -> Tuple[str, Any]:
        """
        Wav2Lip処理とFaceFusion前処理を並列実行
        """
        logger.info("👄 Starting parallel Wav2Lip + FaceFusion preprocessing")

        loop = asyncio.get_event_loop()

        # Wav2Lip処理タスク
        wav2lip_future = loop.run_in_executor(
            self.executor,
            wav2lip_func,
            audio_path,
            video_data
        )

        # FaceFusionモデルロードタスク
        facefusion_prep_future = loop.run_in_executor(
            self.executor,
            facefusion_prep_func,
            source_face
        )

        # 両タスクの完了を待つ
        wav2lip_result, facefusion_prep = await asyncio.gather(
            wav2lip_future,
            facefusion_prep_future
        )

        # GPU メモリクリア
        self.clear_gpu_memory()

        return wav2lip_result, facefusion_prep

    def cleanup(self):
        """リソースのクリーンアップ"""
        self.executor.shutdown(wait=True)
        self.clear_gpu_memory()


class StreamingPipeline:
    """ストリーミング処理パイプライン"""

    def __init__(self):
        self.chunk_size = 1024  # 音声チャンクサイズ
        self.frame_buffer_size = 30  # フレームバッファサイズ

    async def stream_audio_to_video(self,
                                   audio_generator,
                                   video_processor,
                                   output_path: str):
        """
        音声をストリーミングしながら動画処理
        """
        logger.info("🌊 Starting streaming pipeline")

        audio_buffer = queue.Queue(maxsize=10)
        video_buffer = queue.Queue(maxsize=self.frame_buffer_size)

        # 音声ストリーミングスレッド
        def audio_streaming_worker():
            for chunk in audio_generator:
                audio_buffer.put(chunk)
            audio_buffer.put(None)  # 終了シグナル

        # 動画処理スレッド
        def video_processing_worker():
            while True:
                audio_chunk = audio_buffer.get()
                if audio_chunk is None:
                    break

                # 音声チャンクから動画フレーム生成
                frames = video_processor.process_audio_chunk(audio_chunk)
                for frame in frames:
                    video_buffer.put(frame)

                # 定期的にGPUメモリクリア
                if audio_buffer.qsize() % 5 == 0:
                    torch.cuda.empty_cache()

            video_buffer.put(None)  # 終了シグナル

        # スレッド起動
        audio_thread = threading.Thread(target=audio_streaming_worker)
        video_thread = threading.Thread(target=video_processing_worker)

        audio_thread.start()
        video_thread.start()

        # 結果の収集
        output_frames = []
        while True:
            frame = video_buffer.get()
            if frame is None:
                break
            output_frames.append(frame)

        audio_thread.join()
        video_thread.join()

        return output_frames


def optimize_gpu_usage(func):
    """
    GPUメモリ最適化デコレータ
    関数実行前後でGPUメモリをクリア
    """
    def wrapper(*args, **kwargs):
        # 実行前にメモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated()

        result = func(*args, **kwargs)

        # 実行後にメモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            after = torch.cuda.memory_allocated()
            freed = (before - after) / 1024**3
            if freed > 0:
                logger.info(f"🧹 Freed {freed:.2f}GB GPU memory after {func.__name__}")

        return result
    return wrapper


class GPUMemoryManager:
    """
    GPUメモリ管理クラス
    処理ごとに適切なメモリ制限を設定
    """

    def __init__(self):
        self.memory_limits = {
            'llama': 8.0,      # 8GB
            'sovits': 4.0,     # 4GB
            'wav2lip': 3.0,    # 3GB
            'facefusion': 4.0  # 4GB
        }

    def set_memory_limit(self, process_name: str):
        """プロセスごとのメモリ制限設定"""
        if torch.cuda.is_available():
            limit = self.memory_limits.get(process_name, 4.0)
            # PyTorchのメモリ割り当て戦略を設定
            torch.cuda.set_per_process_memory_fraction(limit / 24.0)  # 24GB GPUを想定
            logger.info(f"📏 Set GPU memory limit for {process_name}: {limit}GB")

    def release_memory(self, model_ref=None):
        """モデルとメモリの解放"""
        if model_ref is not None:
            del model_ref

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# 使用例
if __name__ == "__main__":
    # パイプライン最適化のテスト
    optimizer = ParallelPipelineOptimizer()
    memory_manager = GPUMemoryManager()

    # GPUメモリ状態確認
    optimizer.monitor_gpu_memory()

    # メモリクリア
    optimizer.clear_gpu_memory()

    print("✅ Parallel Pipeline Optimizer ready!")