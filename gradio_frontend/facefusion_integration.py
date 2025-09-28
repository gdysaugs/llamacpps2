#!/usr/bin/env python3
"""
FaceFusion統合モジュール（成功したCLIサブプロセス版ベース）
Gradio Frontend用のFaceFusion 順次処理（Face Swap → GFPGAN）統合機能
"""

import subprocess
import sys
import os
import time
import gc
import psutil
import shutil
import fcntl
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class FaceFusionIntegration:
    """FaceFusion統合処理クラス（Face Swap + GFPGAN順次処理）"""

    def __init__(self):
        """初期化"""
        self.wav2lip_root = Path(__file__).parent.parent

        # FaceFusion環境パス (Docker対応)
        if os.path.exists("/app/facefusion_env"):
            # Docker環境
            self.facefusion_env = Path("/app/facefusion_env")
        else:
            # ローカル環境
            self.facefusion_env = self.wav2lip_root / "facefusion_env"
        self.facefusion_script = self.wav2lip_root / "facefusion/facefusion.py"
        self.temp_dir = Path("/tmp/gradio_facefusion")
        self.temp_dir.mkdir(exist_ok=True)

        # サポートする画像・動画形式
        self.supported_image_formats = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        self.supported_video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    def setup_cuda_environment(self) -> str:
        """CUDA環境変数を設定"""
        venv_lib = self.facefusion_env / "lib/python3.10/site-packages"
        cuda_paths = [
            venv_lib / "nvidia/cuda_runtime/lib",
            venv_lib / "nvidia/cudnn/lib",
            venv_lib / "nvidia/cublas/lib",
            venv_lib / "nvidia/cufft/lib"
        ]

        ld_library_path = ":".join(str(p) for p in cuda_paths)
        current_path = os.environ.get("LD_LIBRARY_PATH", "")
        if current_path:
            ld_library_path = f"{ld_library_path}:{current_path}"

        return ld_library_path

    def get_memory_info(self) -> Tuple[float, float]:
        """現在のメモリ使用状況を取得"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        gpu_memory = self.get_gpu_memory()
        return memory_mb, gpu_memory

    def get_gpu_memory(self) -> float:
        """GPU メモリ使用量を取得"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except:
            pass
        return 0

    def clear_memory(self):
        """メモリを明示的に解放"""
        # Python ガベージコレクション
        gc.collect()

        # GPU メモリクリア
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        except Exception:
            pass

        # 再度ガベージコレクション
        gc.collect()

    def monitor_process(self, process, step_name, timeout=600) -> Tuple[bool, float, float]:
        """プロセス監視とメモリ使用量チェック（改良版）"""
        start_time = time.time()
        max_memory = 0
        output_lines = []
        last_output_time = start_time

        while process.poll() is None:
            try:
                # プロセスの出力を読み取る（ノンブロッキング）
                if process.stdout:
                    try:
                        # ファイルディスクリプタをノンブロッキングに設定
                        fd = process.stdout.fileno()
                        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                    except (OSError, AttributeError):
                        # ノンブロッキング設定に失敗した場合はそのまま続行
                        pass

                    try:
                        line = process.stdout.readline()
                        if line and line.strip():
                            output_lines.append(line.strip())
                            print(f"[{step_name}] {line.strip()}")
                            last_output_time = time.time()
                    except (BlockingIOError, OSError):
                        # ノンブロッキング読み取りで何もない場合
                        pass

                # メモリ使用量監視
                try:
                    proc_info = psutil.Process(process.pid)
                    memory_mb = proc_info.memory_info().rss / 1024 / 1024
                    max_memory = max(max_memory, memory_mb)
                except psutil.NoSuchProcess:
                    break

                # タイムアウトチェック
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print(f"[{step_name}] ⏰ タイムアウト ({timeout}秒) - プロセス終了")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return False, max_memory, elapsed

                # 出力がない場合の無応答チェック（300秒）
                if time.time() - last_output_time > 300:
                    print(f"[{step_name}] ⚠️ 300秒間出力なし - 処理継続中...")
                    last_output_time = time.time()  # リセットして継続

            except Exception as e:
                print(f"[{step_name}] 監視エラー: {e}")
                break

            time.sleep(0.5)  # CPU使用率を下げるため少し長めに

        # 残りの出力を読み取る
        if process.stdout:
            remaining_output = process.stdout.read()
            if remaining_output:
                for line in remaining_output.strip().split('\n'):
                    if line.strip():
                        output_lines.append(line.strip())
                        print(f"[{step_name}] {line.strip()}")

        elapsed = time.time() - start_time
        success = process.returncode == 0

        print(f"[{step_name}] 完了 - return code: {process.returncode}, 時間: {elapsed:.2f}秒")

        return success, max_memory, elapsed

    def validate_inputs(self, source_image: str, target_video: str) -> Tuple[bool, str]:
        """入力ファイル検証"""
        try:
            # ソース画像検証
            if not source_image or not Path(source_image).exists():
                return False, "❌ ソース画像ファイルが見つかりません"

            image_ext = Path(source_image).suffix.lower()
            if image_ext not in self.supported_image_formats:
                return False, f"❌ サポートされていない画像形式: {image_ext}"

            # ターゲット動画検証
            if not target_video or not Path(target_video).exists():
                return False, "❌ ターゲット動画ファイルが見つかりません"

            video_ext = Path(target_video).suffix.lower()
            if video_ext not in self.supported_video_formats:
                return False, f"❌ サポートされていない動画形式: {video_ext}"

            # FaceFusion環境確認
            if not self.is_available():
                return False, "❌ FaceFusion環境が見つかりません"

            return True, "✅ FaceFusion入力検証OK"

        except Exception as e:
            return False, f"❌ 入力検証エラー: {str(e)}"

    def prepare_input_files(self, source_image: str, target_video: str) -> Tuple[Path, Path]:
        """入力ファイルの準備"""
        # タイムスタンプ付きファイル名
        timestamp = int(time.time())

        # 入力ディレクトリ確認・作成
        input_dir = self.wav2lip_root / "input"
        input_dir.mkdir(exist_ok=True)

        # ソース画像のコピー（既存の場合は上書き）
        source_ext = Path(source_image).suffix
        source_dest = input_dir / f"facefusion_source_{timestamp}{source_ext}"
        shutil.copy2(source_image, source_dest)

        # ターゲット動画のコピー
        video_ext = Path(target_video).suffix
        video_dest = input_dir / f"facefusion_target_{timestamp}{video_ext}"
        shutil.copy2(target_video, video_dest)

        return source_dest, video_dest

    def process_face_swap_with_gfpgan(
        self,
        source_image: str,
        target_video: str,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        FaceFusion顔交換処理（Face Swap → GFPGAN順次処理）

        Args:
            source_image: ソース画像パス
            target_video: ターゲット動画パス
            progress_callback: 進捗コールバック関数

        Returns:
            処理結果辞書
        """
        start_time = time.time()

        try:
            # 進捗更新
            if progress_callback:
                progress_callback(0.0, "🔍 FaceFusion入力ファイル検証中...")

            # 入力検証
            is_valid, message = self.validate_inputs(source_image, target_video)
            if not is_valid:
                return {
                    "success": False,
                    "message": message,
                    "video_path": None,
                    "stats": {}
                }

            # 進捗更新
            if progress_callback:
                progress_callback(0.1, "📁 FaceFusionファイル準備中...")

            # ファイル準備
            source_dest, video_dest = self.prepare_input_files(source_image, target_video)

            # 出力パス設定
            timestamp = int(time.time())
            temp_video = self.wav2lip_root / "output/facefusion" / f"temp_{timestamp}.mp4"
            output_video = self.wav2lip_root / "output/facefusion" / f"gradio_facefusion_{timestamp}.mp4"
            output_video.parent.mkdir(parents=True, exist_ok=True)

            # CUDA環境設定
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = self.setup_cuda_environment()

            # 初期メモリ状況
            initial_mem = self.get_memory_info()

            # Step 1: Face Swapper
            if progress_callback:
                progress_callback(0.2, "🎭 Step 1: 顔交換処理中...")

            # 成功した単体テスト通りのコマンド構成
            cmd1 = [
                "bash", "-c",
                f"cd {self.wav2lip_root / 'facefusion'} && " +
                f"source {self.facefusion_env / 'bin/activate'} && " +
                f"export LD_LIBRARY_PATH='{env['LD_LIBRARY_PATH']}' && " +
                f"export PYTHONPATH='/app/facefusion:$PYTHONPATH' && " +
                f"{self.facefusion_env}/bin/python facefusion.py headless-run " +
                f"--source-paths {source_dest} " +
                f"--target-path {video_dest} " +
                f"--output-path {temp_video} " +
                f"--temp-path /app/models/facefusion " +
                f"--execution-providers cuda " +
                f"--face-detector-model retinaface " +
                f"--face-detector-size 320x320 " +
                f"--face-detector-score 0.3 " +
                f"--face-swapper-model inswapper_128 " +
                f"--processors face_swapper " +
                f"--execution-thread-count 2 " +
                f"--video-memory-strategy tolerant " +
                f"--log-level info"
            ]

            print(f"[Face Swap] コマンド実行中: {' '.join(cmd1[:2])}...")
            process1 = subprocess.Popen(cmd1)

            # bash -cコマンドは直接待機（monitor_processは使わない）
            start_time = time.time()
            return_code = process1.wait()
            elapsed_time1 = time.time() - start_time
            success1 = return_code == 0
            max_memory1 = 0

            # デバッグ情報を追加
            print(f"🔍 DEBUG Face Swap: success1={success1}, return_code={process1.returncode}, temp_video.exists()={temp_video.exists()}")

            if not success1:
                return {
                    "success": False,
                    "message": f"❌ FaceFusion Face Swap失敗 (return_code: {process1.returncode})",
                    "video_path": None,
                    "stats": {"face_swap_time": elapsed_time1}
                }

            # Step 2: GFPGAN Enhancement
            if progress_callback:
                progress_callback(0.6, "✨ Step 2: GFPGAN品質向上中...")

            # 成功した単体テスト通りのコマンド構成
            cmd2 = [
                "bash", "-c",
                f"cd {self.wav2lip_root / 'facefusion'} && " +
                f"source {self.facefusion_env / 'bin/activate'} && " +
                f"export LD_LIBRARY_PATH='{env['LD_LIBRARY_PATH']}' && " +
                f"export PYTHONPATH='/app/facefusion:$PYTHONPATH' && " +
                f"{self.facefusion_env}/bin/python facefusion.py headless-run " +
                f"--source-paths {temp_video} " +
                f"--target-path {temp_video} " +
                f"--output-path {output_video} " +
                f"--temp-path /app/models/facefusion " +
                f"--execution-providers cuda " +
                f"--processors face_enhancer " +
                f"--face-enhancer-model gfpgan_1.4 " +
                f"--face-enhancer-blend 25 " +
                f"--face-enhancer-weight 0.5 " +
                f"--execution-thread-count 2 " +
                f"--video-memory-strategy tolerant " +
                f"--log-level info"
            ]

            print(f"[GFPGAN Enhancement] コマンド実行中: {' '.join(cmd2[:2])}...")
            process2 = subprocess.Popen(cmd2)

            # bash -cコマンドは直接待機（monitor_processは使わない）
            start_time2 = time.time()
            return_code2 = process2.wait()
            elapsed_time2 = time.time() - start_time2
            success2 = return_code2 == 0
            max_memory2 = 0

            # デバッグ情報を追加
            print(f"🔍 DEBUG GFPGAN: success2={success2}, return_code={process2.returncode}, output_video.exists()={output_video.exists()}")

            if not success2:
                return {
                    "success": False,
                    "message": f"❌ FaceFusion GFPGAN Enhancement失敗 (return_code: {process2.returncode})",
                    "video_path": None,
                    "stats": {
                        "face_swap_time": elapsed_time1,
                        "gfpgan_time": elapsed_time2
                    }
                }

            # Step 3: 音声復元処理
            if progress_callback:
                progress_callback(0.85, "🎵 音声トラック復元中...")

            final_output = self.wav2lip_root / "output/facefusion" / f"gradio_facefusion_with_audio_{timestamp}.mp4"
            success3 = self.restore_audio_track(str(video_dest), str(output_video), str(final_output))

            if success3:
                output_video = final_output

            # 処理完了後のメモリ解放
            if progress_callback:
                progress_callback(0.9, "🧹 メモリクリーンアップ中...")

            self.clear_memory()

            # メモリ解放後の状況確認
            final_mem = self.get_memory_info()

            # 全体の結果
            total_elapsed = time.time() - start_time
            max_memory = max(max_memory1, max_memory2)

            if progress_callback:
                progress_callback(1.0, "✅ FaceFusion処理完了!")

            # 統計情報
            stats = {
                "face_swap_time": elapsed_time1,
                "gfpgan_time": elapsed_time2,
                "audio_restore": success3 if 'success3' in locals() else False,
                "total_time": total_elapsed,
                "max_memory_mb": max_memory,
                "initial_memory": initial_mem,
                "final_memory": final_mem
            }

            # 一時ファイルクリーンアップ
            self.cleanup_temp_files([source_dest, video_dest, temp_video])

            # デバッグ情報を追加
            print(f"🔍 DEBUG Final: output_video={output_video}")
            print(f"🔍 DEBUG Final: output_video.exists()={output_video.exists()}")
            if success3 and 'final_output' in locals():
                print(f"🔍 DEBUG Final: final_output={final_output}")
                print(f"🔍 DEBUG Final: final_output.exists()={final_output.exists() if final_output else 'N/A'}")

            if output_video.exists():
                size_mb = output_video.stat().st_size / (1024 * 1024)
                stats["output_size_mb"] = size_mb

                audio_msg = " + 音声復元" if success3 else ""
                message = f"✅ FaceFusion完了: Face Swap {elapsed_time1:.2f}秒 + GFPGAN {elapsed_time2:.2f}秒{audio_msg} = 合計 {total_elapsed:.2f}秒"

                print(f"🔍 DEBUG Final: returning SUCCESS with video_path={str(output_video)}")
                return {
                    "success": True,
                    "message": message,
                    "video_path": str(output_video),
                    "stats": stats
                }
            else:
                print(f"🔍 DEBUG Final: returning FAILURE - output file does not exist")
                return {
                    "success": False,
                    "message": "❌ FaceFusion出力ファイルが作成されませんでした",
                    "video_path": None,
                    "stats": stats
                }

        except Exception as e:
            total_elapsed = time.time() - start_time
            return {
                "success": False,
                "message": f"❌ FaceFusion実行エラー: {str(e)}",
                "video_path": None,
                "stats": {"total_time": total_elapsed}
            }

    def restore_audio_track(self, original_video: str, processed_video: str, output_video: str) -> bool:
        """
        FaceFusion処理後のビデオに元の音声トラックを復元

        Args:
            original_video: 元の動画（音声付き）
            processed_video: FaceFusion処理済み動画（音声なし）
            output_video: 出力動画パス（音声付き）

        Returns:
            成功: True, 失敗: False
        """
        try:
            import subprocess

            # ffmpegで音声トラックを復元
            cmd = [
                "ffmpeg",
                "-i", processed_video,  # ビデオソース（FaceFusion処理済み）
                "-i", original_video,   # オーディオソース（元動画）
                "-c:v", "copy",         # ビデオコーデックをコピー
                "-c:a", "aac",          # オーディオをAACで再エンコード
                "-map", "0:v:0",        # 最初のファイルのビデオトラック
                "-map", "1:a:0",        # 二番目のファイルのオーディオトラック
                "-shortest",            # 短い方に合わせる
                "-y",                   # 上書き確認なし
                output_video
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            process.wait()

            return process.returncode == 0 and Path(output_video).exists()

        except Exception as e:
            print(f"音声復元エラー: {e}")
            return False

    def cleanup_temp_files(self, file_paths):
        """一時ファイルのクリーンアップ"""
        for file_path in file_paths:
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
            except Exception:
                pass  # サイレントに失敗を無視

    def is_available(self) -> bool:
        """FaceFusion機能が利用可能かチェック"""
        return (
            self.facefusion_script.exists() and
            self.facefusion_env.exists() and
            (self.facefusion_env / "bin/python").exists()
        )