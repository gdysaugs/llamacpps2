#!/usr/bin/env python3
"""
SOVITS-Wav2Lip統合処理システム
ボイスクローン音声生成 → メモリ解放 → 口パク動画生成 → メモリ解放
"""
import subprocess
import sys
import os
import time
import tempfile
import gc
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# 無音削減機能をインポート
from audio_silence_reducer import reduce_audio_silence

class SOVITSWav2LipIntegration:
    def __init__(self):
        """統合処理システムの初期化"""
        self.project_root = Path(__file__).parent.parent

        # サブプロセススクリプトパス (Docker対応)
        if os.path.exists("/app/gpt_sovits_full/gpt_sovits_simple_cli.py"):
            # Docker環境
            self.sovits_script = Path("/app/gpt_sovits_full/gpt_sovits_simple_cli.py")
            self.wav2lip_script = Path("/app/wav2lip_subprocess.py")
        else:
            # ローカル環境
            self.sovits_script = self.project_root / "gpt_sovits_simple_cli.py"
            self.wav2lip_script = self.project_root / "wav2lip_subprocess.py"

        # 仮想環境パス (Docker対応)
        if os.path.exists("/app/sovits_venv"):
            # Docker環境（sovits_venv）
            self.sovits_venv = Path("/app/sovits_venv/bin/python")
            self.wav2lip_venv = Path("/app/venv/bin/python")
        elif os.path.exists("/app/gpt_sovits_env"):
            # Docker環境（gpt_sovits_env - 別パターン）
            self.sovits_venv = Path("/app/gpt_sovits_env/bin/python")
            self.wav2lip_venv = Path("/app/venv/bin/python")
        else:
            # ローカル環境
            self.sovits_venv = self.project_root / "gpt_sovits_env" / "bin" / "python"
            self.wav2lip_venv = self.project_root / "venv" / "bin" / "python"

        # 一時ディレクトリ (Docker対応)
        if os.path.exists("/app"):
            # Docker環境
            self.temp_dir = Path("/app/temp/sovits_wav2lip_integration")
        else:
            # ローカル環境
            self.temp_dir = Path("/tmp/sovits_wav2lip_integration")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # 検証
        self._verify_setup()

    def _verify_setup(self):
        """セットアップ検証"""
        missing = []

        if not self.sovits_script.exists():
            missing.append(f"SOVITS script: {self.sovits_script}")
        if not self.wav2lip_script.exists():
            missing.append(f"Wav2Lip script: {self.wav2lip_script}")
        if not self.sovits_venv.exists():
            missing.append(f"SOVITS venv: {self.sovits_venv}")
        if not self.wav2lip_venv.exists():
            missing.append(f"Wav2Lip venv: {self.wav2lip_venv}")

        if missing:
            raise FileNotFoundError(f"Missing components:\n" + "\n".join(missing))

    def _generate_temp_filename(self, suffix: str) -> str:
        """一時ファイル名生成"""
        timestamp = int(time.time() * 1000)
        return f"temp_integration_{timestamp}{suffix}"

    def run_sovits_voice_clone(
        self,
        script_text: str,
        reference_audio_path: str,
        speed_factor: float = 1.0
    ) -> Tuple[Optional[str], str, Dict[str, Any]]:
        """
        Phase 1: SOVITSボイスクローン音声生成

        Args:
            script_text: 生成したいセリフテキスト
            reference_audio_path: リファレンス音声ファイルパス

        Returns:
            Tuple[Optional[str], str, Dict]: (生成音声パス, ログ, 統計)
        """
        try:
            print("=" * 60)
            print("🎭 Phase 1: SOVITS Voice Cloning Started")
            print("=" * 60)

            # 出力ファイル名を生成（gpt_sovits_simple_cli.pyはoutput/ディレクトリに保存）
            output_filename = self._generate_temp_filename(".wav")

            # リファレンス音声の存在確認と安定した場所へのコピー
            import shutil
            import hashlib
            from pathlib import Path

            # Gradioファイルパスの処理を改善
            if hasattr(reference_audio_path, 'name'):
                actual_path = reference_audio_path.name
            elif isinstance(reference_audio_path, str):
                actual_path = reference_audio_path
            else:
                actual_path = str(reference_audio_path)

            reference_path = Path(actual_path)
            print(f"🔍 参照音声パスチェック: {reference_path}")
            print(f"🔍 ファイル存在確認: {reference_path.exists()}")

            if not reference_path.exists():
                raise FileNotFoundError(f"❌ 参照音声ファイルが見つかりません: {reference_path}")

            # 安全なファイル名でコピー
            safe_audio_name = f"ref_audio_{hashlib.md5(str(actual_path).encode()).hexdigest()[:8]}.mp3"
            safe_audio_path = self.temp_dir / safe_audio_name

            # 一時ディレクトリが存在することを確認
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 一時ディレクトリ確認: {self.temp_dir}")

            # 毎回新しくコピーして確実性を保つ
            try:
                shutil.copy2(reference_path, safe_audio_path)
                print(f"📁 音声ファイルをコピー完了: {reference_path} -> {safe_audio_path}")
                print(f"📊 コピー後ファイルサイズ: {safe_audio_path.stat().st_size} bytes")
            except Exception as copy_error:
                print(f"❌ コピーエラー詳細: {copy_error}")
                print(f"❌ ソースパス: {reference_path} (存在: {reference_path.exists()})")
                print(f"❌ 宛先パス: {safe_audio_path}")
                print(f"❌ 宛先ディレクトリ: {safe_audio_path.parent} (存在: {safe_audio_path.parent.exists()})")
                raise Exception(f"❌ ファイルコピーエラー: {copy_error}")

            # SOVITSコマンド構築（gpt_sovits_simple_cli.pyの引数形式）
            # python gpt_sovits_simple_cli.py "参照音声" "生成テキスト" "出力ファイル名" --speed
            cmd = [
                str(self.sovits_venv),
                str(self.sovits_script),
                str(safe_audio_path),        # 安定した参照音声パス
                script_text,                 # 生成したいテキスト
                output_filename,             # 出力ファイル名（output/に保存される）
                "--speed", str(speed_factor) # 速度パラメータ
            ]

            # 実際の出力パスは output/ ディレクトリ (Docker対応)
            if os.path.exists("/app/output"):
                # Docker環境
                actual_output_path = Path("/app/output") / output_filename
            else:
                # ローカル環境
                actual_output_path = self.project_root / "output" / output_filename

            print(f"📝 Script Text: {script_text[:50]}..." if len(script_text) > 50 else f"📝 Script Text: {script_text}")
            print(f"🎵 Reference Audio: {actual_path}")
            print(f"💾 Output Path: {actual_output_path}")
            print(f"📋 Command: {' '.join(cmd)}")
            print(f"⚙️ Using GPT-SoVITS v4 with enhanced emotion (temperature=2.0)")

            # サブプロセス実行
            start_time = time.time()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5分タイムアウト
                cwd=str(self.project_root)
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # 結果処理
            initial_success = result.returncode == 0

            print("=" * 60)
            print(f"💥 Return code: {result.returncode}")
            print(f"⏱️ Execution time: {execution_time:.2f} seconds")
            print("=" * 60)

            # 詳細ログ出力（成功メッセージを探す）
            full_log = ""
            success_indicators = ["✅ ボイスクローン成功", "✅ ボイスクローン完了", "ボイスクローン完了"]
            generation_successful = False

            if result.stdout:
                print("📄 SOVITS Output:")
                print(result.stdout)
                full_log += f"STDOUT:\n{result.stdout}\n\n"
                # 成功インジケータをチェック
                for indicator in success_indicators:
                    if indicator in result.stdout:
                        generation_successful = True
                        break

            if result.stderr:
                # stderrにはログ情報が含まれる（エラーではない場合もある）
                if "ERROR" in result.stderr or "error" in result.stderr:
                    print("⚠️ SOVITS Warnings/Errors:")
                    print(result.stderr)
                full_log += f"STDERR:\n{result.stderr}\n\n"

            # ファイル存在確認（returncode=0でもファイルが無い場合がある）
            file_found = actual_output_path.exists()
            file_size = 0
            if file_found:
                file_size = actual_output_path.stat().st_size
                print(f"📦 Output file found: {actual_output_path} ({file_size} bytes)")
            else:
                print(f"❌ Output file not found: {actual_output_path}")

            # 成功判定：(returncode=0 または 成功メッセージあり) かつ ファイルが存在 かつ サイズ>1KB
            success = (initial_success or generation_successful) and file_found and file_size > 1000

            if success:
                print("✅ SOVITS Voice Cloning Completed!")
            else:
                print("❌ SOVITS Voice Cloning Failed!")
                if initial_success and not file_found:
                    print("   -> Command succeeded but output file not found")
                elif initial_success and file_size <= 1000:
                    print(f"   -> File too small: {file_size} bytes")

            # 統計情報
            stats = {
                "success": success,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "script_text": script_text,
                "output_size_mb": file_size / (1024 * 1024) if file_size > 0 else 0,
                "file_found": file_found,
                "file_size": file_size
            }

            if success:
                # 🔇 無音削減処理を実行
                silence_reduction_message = ""
                try:
                    print("🔇 無音削減処理開始...")
                    silence_success, silence_msg, processed_path = reduce_audio_silence(
                        str(actual_output_path),
                        max_silence_duration=1.0
                    )
                    if silence_success:
                        silence_reduction_message = f"\n🔇 無音削減: {silence_msg}"
                        stats["silence_reduction"] = True
                        stats["silence_reduction_message"] = silence_msg
                    else:
                        silence_reduction_message = f"\n⚠️ 無音削減失敗: {silence_msg}"
                        stats["silence_reduction"] = False
                        stats["silence_reduction_error"] = silence_msg
                except Exception as e:
                    silence_reduction_message = f"\n❌ 無音削減エラー: {str(e)}"
                    stats["silence_reduction"] = False
                    stats["silence_reduction_error"] = str(e)

                log_message = f"""🎭 SOVITS Voice Cloning Success!
⏱️ Processing time: {execution_time:.2f}s
📝 Script: {script_text[:100]}...
💾 Output: {output_filename}
📦 File size: {stats['output_size_mb']:.2f}MB
🎵 Voice quality: High (GPT-SoVITS v4){silence_reduction_message}

📄 Process Output:
{full_log}
"""
                return str(actual_output_path), log_message, stats
            else:
                # エラー診断
                error_reason = ""
                if not initial_success:
                    error_reason = f"Process failed with return code {result.returncode}"
                elif not file_found:
                    error_reason = "Output file was not created (check if SOVITS processed correctly)"
                elif file_size <= 1000:
                    error_reason = f"Output file too small ({file_size} bytes)"

                error_msg = f"""❌ SOVITS Voice Cloning Failed
💥 Return code: {result.returncode}
⏱️ Execution time: {execution_time:.2f}s
📝 Script: {script_text[:100]}...
🔍 Error reason: {error_reason}

📄 Process Output:
{full_log}

📄 Diagnostics:
- Command executed: {' '.join(cmd)}
- Expected output: {actual_output_path}
- File exists: {file_found}
- File size: {file_size} bytes
"""
                return None, error_msg, stats

        except subprocess.TimeoutExpired:
            return None, "❌ SOVITS processing timed out (5 minutes)", {"success": False, "error": "timeout"}
        except Exception as e:
            return None, f"❌ SOVITS processing error: {str(e)}", {"success": False, "error": str(e)}

    def _force_memory_cleanup(self):
        """強制メモリクリーンアップ"""
        print("🧹 Forcing memory cleanup...")
        gc.collect()
        # 追加のクリーンアップが必要な場合はここに
        time.sleep(2)  # プロセス終了待機
        print("✅ Memory cleanup completed")

    def run_wav2lip_with_cloned_voice(
        self,
        video_path: str,
        cloned_audio_path: str,
        use_gfpgan: bool = True,
        device: str = "cuda"
    ) -> Tuple[Optional[str], str, Dict[str, Any]]:
        """
        Phase 2: Wav2Lip口パク動画生成（クローン音声使用）

        Args:
            video_path: 入力動画パス
            cloned_audio_path: Phase1で生成したクローン音声パス
            use_gfpgan: GFPGAN使用フラグ
            device: 処理デバイス

        Returns:
            Tuple[Optional[str], str, Dict]: (出力動画パス, ログ, 統計)
        """
        try:
            print("=" * 60)
            print("🎬 Phase 2: Wav2Lip Lip Sync Started")
            print("=" * 60)

            # 出力ファイル設定
            output_filename = self._generate_temp_filename(".mp4")
            output_path = self.temp_dir / output_filename

            # Wav2Lipサブプロセスラッパーコマンド構築（30分タイムアウト付き）
            subprocess_script = self.project_root / "wav2lip_subprocess.py"
            cmd = [
                sys.executable,
                str(subprocess_script),
                video_path,
                cloned_audio_path,
                "-o", str(output_path),
                "--device", device
            ]

            if not use_gfpgan:
                cmd.append("--no-gfpgan")

            print(f"📹 Video: {Path(video_path).name}")
            print(f"🎵 Cloned Audio: {Path(cloned_audio_path).name}")
            print(f"💾 Output: {output_filename}")
            print(f"⚙️ GFPGAN: {'Enabled' if use_gfpgan else 'Disabled'}")
            print(f"💻 Device: {device.upper()}")

            # サブプロセス実行
            start_time = time.time()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2100,  # 35分タイムアウト（サブプロセスラッパー内の30分 + 余裕5分）
                cwd=str(self.project_root)
            )

            end_time = time.time()
            execution_time = end_time - start_time

            success = result.returncode == 0

            print("=" * 60)
            if success:
                print("✅ Wav2Lip Lip Sync Completed!")
            else:
                print("❌ Wav2Lip Lip Sync Failed!")
            print(f"⏱️ Execution time: {execution_time:.2f} seconds")
            print(f"💥 Return code: {result.returncode}")
            print("=" * 60)

            # ログ出力（パフォーマンス情報抽出）
            if result.stdout:
                print("📄 Wav2Lip Stdout:")
                print(result.stdout[-1000:])
            if result.stderr:
                print("📄 Wav2Lip Stderr:")
                print(result.stderr[-500:])

            # 統計情報
            stats = {
                "success": success,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "use_gfpgan": use_gfpgan,
                "device": device,
                "output_size_mb": 0
            }

            if success and output_path.exists():
                # ファイルサイズ取得
                file_size = output_path.stat().st_size
                stats["output_size_mb"] = file_size / (1024 * 1024)

                log_message = f"""🎬 Wav2Lip Lip Sync Success!
⏱️ Processing time: {execution_time:.2f}s
💾 Output: {output_filename}
📦 File size: {stats['output_size_mb']:.2f}MB
⚙️ GFPGAN: {'Enabled' if use_gfpgan else 'Disabled'}
💻 Device: {device.upper()}
🎥 Quality: High (RetinaFace + ONNX GPU)
"""
                return str(output_path), log_message, stats
            else:
                error_msg = f"""❌ Wav2Lip Lip Sync Failed
💥 Return code: {result.returncode}
⏱️ Execution time: {execution_time:.2f}s

📄 Error details:
{result.stderr if result.stderr else 'No error details available'}
"""
                return None, error_msg, stats

        except subprocess.TimeoutExpired:
            return None, "❌ Wav2Lip processing timed out (35 minutes)", {"success": False, "error": "timeout"}
        except Exception as e:
            return None, f"❌ Wav2Lip processing error: {str(e)}", {"success": False, "error": str(e)}

    def run_integrated_pipeline(
        self,
        video_path: str,
        reference_audio_path: str,
        script_text: str,
        use_gfpgan: bool = True,
        device: str = "cuda"
    ) -> Tuple[Optional[str], str, Dict[str, Any]]:
        """
        統合パイプライン実行
        Phase 1: SOVITS ボイスクローン → メモリ解放
        Phase 2: Wav2Lip 口パク生成 → メモリ解放

        Args:
            video_path: 入力動画ファイル
            reference_audio_path: リファレンス音声ファイル
            script_text: 生成したいセリフテキスト
            use_gfpgan: GFPGAN使用フラグ
            device: 処理デバイス

        Returns:
            Tuple[Optional[str], str, Dict]: (最終出力動画パス, 統合ログ, 統合統計)
        """
        print("🚀" * 20)
        print("🎭🎬 SOVITS-Wav2Lip Integration Pipeline Started")
        print("🚀" * 20)

        # 統合統計情報
        pipeline_start_time = time.time()
        integrated_stats = {
            "pipeline_success": False,
            "total_execution_time": 0,
            "phase1_stats": {},
            "phase2_stats": {},
            "script_text": script_text,
            "settings": {
                "use_gfpgan": use_gfpgan,
                "device": device
            }
        }

        integrated_log = ""

        try:
            # Phase 1: SOVITS Voice Cloning
            print("\n🎭 Starting Phase 1: SOVITS Voice Cloning...")

            cloned_audio_path, phase1_log, phase1_stats = self.run_sovits_voice_clone(
                script_text=script_text,
                reference_audio_path=reference_audio_path
            )

            integrated_log += f"【Phase 1: SOVITS Voice Cloning】\n{phase1_log}\n\n"
            integrated_stats["phase1_stats"] = phase1_stats

            if not phase1_stats.get("success", False):
                integrated_log += "❌ Phase 1 failed. Pipeline stopped.\n"
                return None, integrated_log, integrated_stats

            print("✅ Phase 1 completed successfully!")

            # メモリクリーンアップ
            self._force_memory_cleanup()

            # Phase 2: Wav2Lip Lip Sync
            print("\n🎬 Starting Phase 2: Wav2Lip Lip Sync...")

            final_video_path, phase2_log, phase2_stats = self.run_wav2lip_with_cloned_voice(
                video_path=video_path,
                cloned_audio_path=cloned_audio_path,
                use_gfpgan=use_gfpgan,
                device=device
            )

            integrated_log += f"【Phase 2: Wav2Lip Lip Sync】\n{phase2_log}\n\n"
            integrated_stats["phase2_stats"] = phase2_stats

            if not phase2_stats.get("success", False):
                integrated_log += "❌ Phase 2 failed. Pipeline stopped.\n"
                return None, integrated_log, integrated_stats

            print("✅ Phase 2 completed successfully!")

            # 最終メモリクリーンアップ
            self._force_memory_cleanup()

            # パイプライン完了
            pipeline_end_time = time.time()
            total_time = pipeline_end_time - pipeline_start_time

            integrated_stats["pipeline_success"] = True
            integrated_stats["total_execution_time"] = total_time

            # 最終サマリー
            summary_log = f"""
🎉 SOVITS-Wav2Lip Integration Pipeline Completed!

📊 Pipeline Summary:
⏱️ Total execution time: {total_time:.2f}s
📝 Script text: {script_text[:100]}...
🎭 Phase 1 (SOVITS): {phase1_stats.get('execution_time', 0):.2f}s
🎬 Phase 2 (Wav2Lip): {phase2_stats.get('execution_time', 0):.2f}s

📦 Output Details:
🎵 Cloned audio size: {phase1_stats.get('output_size_mb', 0):.2f}MB
🎥 Final video size: {phase2_stats.get('output_size_mb', 0):.2f}MB
⚙️ GFPGAN: {'Enabled' if use_gfpgan else 'Disabled'}
💻 Device: {device.upper()}

✨ Result: High-quality voice-cloned lip-sync video generated!
"""

            integrated_log += summary_log

            print("🎉" * 20)
            print("🎭🎬 SOVITS-Wav2Lip Integration Completed Successfully!")
            print("🎉" * 20)

            return final_video_path, integrated_log, integrated_stats

        except Exception as e:
            pipeline_end_time = time.time()
            total_time = pipeline_end_time - pipeline_start_time

            integrated_stats["total_execution_time"] = total_time

            error_log = f"""
❌ Pipeline Integration Error!
⏱️ Total time before error: {total_time:.2f}s
💥 Error: {str(e)}

{integrated_log}
"""

            return None, error_log, integrated_stats

    def process_sovits_audio_generation(
        self,
        reference_audio_path: str,
        script_text: str,
        device: str = "cuda",
        progress_callback=None,
        speed_factor: float = 1.0
    ) -> Dict[str, Any]:
        """
        Phase 1用: SoVITS音声生成処理

        Args:
            reference_audio_path: リファレンス音声パス
            script_text: セリフテキスト
            device: 処理デバイス（未使用）
            progress_callback: 進捗コールバック関数

        Returns:
            Dict: 処理結果辞書
        """
        try:
            if progress_callback:
                progress_callback(0.1, "SoVITS音声生成開始...")

            # 既存のrun_sovits_voice_cloneメソッドを使用
            audio_path, log_message, stats = self.run_sovits_voice_clone(
                script_text=script_text,
                reference_audio_path=reference_audio_path,
                speed_factor=speed_factor
            )

            if progress_callback:
                progress_callback(1.0, "SoVITS音声生成完了")

            # 成功判定
            success = audio_path is not None and stats.get("success", False)

            return {
                "success": success,
                "audio_path": audio_path,
                "message": log_message,
                "stats": stats
            }

        except Exception as e:
            return {
                "success": False,
                "audio_path": None,
                "message": f"SoVITS音声生成エラー: {str(e)}",
                "stats": {"error": str(e)}
            }

    def process_wav2lip_lipsync(
        self,
        video_path: str,
        audio_path: str,
        use_gfpgan: bool = True,
        device: str = "cuda",
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Phase 3用: Wav2Lipリップシンク処理

        Args:
            video_path: 動画ファイルパス
            audio_path: 音声ファイルパス
            use_gfpgan: GFPGAN使用フラグ
            device: 処理デバイス
            progress_callback: 進捗コールバック関数

        Returns:
            Dict: 処理結果辞書
        """
        try:
            if progress_callback:
                progress_callback(0.1, "Wav2Lipリップシンク開始...")

            # 既存のrun_wav2lip_with_cloned_voiceメソッドを使用
            video_result_path, log_message, stats = self.run_wav2lip_with_cloned_voice(
                video_path=video_path,
                cloned_audio_path=audio_path,
                use_gfpgan=use_gfpgan,
                device=device
            )

            if progress_callback:
                progress_callback(1.0, "Wav2Lipリップシンク完了")

            # 成功判定
            success = video_result_path is not None and stats.get("success", False)

            return {
                "success": success,
                "video_path": video_result_path,
                "message": log_message,
                "stats": stats
            }

        except Exception as e:
            return {
                "success": False,
                "video_path": None,
                "message": f"Wav2Lipリップシンクエラー: {str(e)}",
                "stats": {"error": str(e)}
            }

    def process_wav2lip_sync(
        self,
        video_path: str,
        audio_path: str,
        use_gfpgan: bool = True,
        device: str = "cuda",
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Phase 2用: Wav2Lipリップシンク処理（Gradio統合用）

        Args:
            video_path: 動画ファイルパス
            audio_path: 音声ファイルパス
            use_gfpgan: GFPGAN使用フラグ
            device: 処理デバイス
            progress_callback: 進捗コールバック関数

        Returns:
            Dict: 処理結果辞書
        """
        # process_wav2lip_lipsyncと同じ処理を実行
        return self.process_wav2lip_lipsync(
            video_path=video_path,
            audio_path=audio_path,
            use_gfpgan=use_gfpgan,
            device=device,
            progress_callback=progress_callback
        )

    def cleanup_temp_files(self):
        """一時ファイルクリーンアップ"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
                print("🧹 Temporary files cleaned up")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")


# 使用例とテスト関数
def quick_test_integration():
    """統合システムのクイックテスト"""
    integration = SOVITSWav2LipIntegration()

    # テスト用のダミーパラメータ（実際のファイルパスに変更してください）
    test_params = {
        "video_path": "/tmp/test_video.mp4",  # 実際のビデオファイルパス
        "reference_audio_path": "/tmp/test_ref.wav",  # 実際のリファレンス音声
        "script_text": "これはテスト用のセリフです。ボイスクローン技術により、リファレンス音声の声質で話します。",
        "use_gfpgan": True,
        "device": "cuda"
    }

    result_video, log, stats = integration.run_integrated_pipeline(**test_params)

    print("=" * 60)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(log)
    print(f"📈 Stats: {stats}")

    if result_video:
        print(f"✅ Test completed! Final video: {result_video}")
    else:
        print("❌ Test failed!")

    return result_video, log, stats


if __name__ == "__main__":
    # コマンドライン実行用
    print("🎭🎬 SOVITS-Wav2Lip Integration System")
    print("Usage: python sovits_wav2lip_integration.py")
    print("For testing, modify the quick_test_integration() function with actual file paths.")

    # テスト実行（ファイルパスを実際のものに変更してから実行）
    # quick_test_integration()