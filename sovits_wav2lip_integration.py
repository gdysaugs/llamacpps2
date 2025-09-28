#!/usr/bin/env python3
"""
SOVITS-Wav2Lip 統合システム
音声クローン生成 → Wav2Lip 口パク動画生成の統合パイプライン
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path

class SOVITSWav2LipIntegration:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root or Path(__file__).parent)

        # パス設定（修正版）
        self.sovits_venv = self.project_root / "gpt_sovits_env" / "bin" / "python"
        self.wav2lip_venv = self.project_root / "gradio_frontend" / "gradio_venv" / "bin" / "python"

        self.sovits_script = self.project_root / "sovits_subprocess_simple.py"
        self.wav2lip_script = self.project_root / "wav2lip_subprocess.py"

        # 出力ディレクトリ
        self.output_dir = self.project_root / "output"
        self.output_dir.mkdir(exist_ok=True)

        print(f"統合システム初期化:")
        print(f"  プロジェクトルート: {self.project_root}")
        print(f"  SOVITS環境: {self.sovits_venv}")
        print(f"  Wav2Lip環境: {self.wav2lip_venv}")

    def _generate_temp_filename(self, suffix):
        """一時ファイル名生成"""
        timestamp = int(time.time())
        return f"temp_audio_{timestamp}{suffix}"

    def run_sovits_voice_clone(self, script_text, ref_audio_path, output_filename=None):
        """SOVITS音声クローン実行"""
        print("=" * 60)
        print("🎤 SOVITS音声クローン開始")
        print("=" * 60)

        try:
            # 出力ファイル設定（sovits_subprocess_simple.pyはoutput/ディレクトリに保存）
            if output_filename is None:
                output_filename = self._generate_temp_filename(".wav")

            print(f"スクリプト: {script_text[:50]}...")
            print(f"リファレンス音声: {ref_audio_path}")
            print(f"出力ファイル名: {output_filename}")

            # SOVITSコマンド構築（Pythonから直接実行）
            cmd = [
                str(self.sovits_venv),
                str(self.sovits_script),
                script_text,
                output_filename  # ファイル名のみ（パスではない）
            ]

            print(f"実行コマンド: {' '.join(cmd)}")
            print("-" * 40)

            start_time = time.time()

            # サブプロセス実行
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5分でタイムアウト
            )

            execution_time = time.time() - start_time

            # 実際の出力パスは output/ ディレクトリ
            actual_output_path = self.project_root / "output" / output_filename

            print(f"\\n実行時間: {execution_time:.2f}秒")
            print(f"リターンコード: {result.returncode}")

            if result.stdout:
                print("STDOUT:")
                print(result.stdout)

            if result.stderr:
                print("STDERR:")
                print(result.stderr)

            if result.returncode == 0:
                # ファイルの存在と情報確認
                if actual_output_path.exists():
                    file_size = actual_output_path.stat().st_size
                    print(f"✅ SOVITS音声生成成功!")
                    print(f"   出力: {actual_output_path}")
                    print(f"   ファイルサイズ: {file_size / 1024 / 1024:.2f}MB")
                    return str(actual_output_path)
                else:
                    print(f"❌ 音声ファイルが生成されませんでした: {actual_output_path}")
                    return None
            else:
                print(f"❌ SOVITS実行エラー (code {result.returncode})")
                return None

        except subprocess.TimeoutExpired:
            print("❌ SOVITS実行タイムアウト (5分)")
            return None
        except Exception as e:
            print(f"❌ SOVITS実行中にエラー: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_wav2lip_with_cloned_voice(self, video_path, cloned_audio_path, output_filename=None):
        """Wav2Lip口パク動画生成"""
        print("=" * 60)
        print("🎬 Wav2Lip口パク動画生成開始")
        print("=" * 60)

        try:
            if output_filename is None:
                timestamp = int(time.time())
                output_filename = f"integrated_result_{timestamp}.mp4"

            output_path = self.output_dir / output_filename

            print(f"入力動画: {video_path}")
            print(f"クローン音声: {cloned_audio_path}")
            print(f"出力動画: {output_path}")

            # Wav2Lipコマンド構築
            cmd = [
                str(self.wav2lip_venv),
                str(self.wav2lip_script),
                str(video_path),
                str(cloned_audio_path),
                "--output", str(output_path)
            ]

            print(f"実行コマンド: {' '.join(cmd)}")
            print("-" * 40)

            start_time = time.time()

            # サブプロセス実行
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=600  # 10分でタイムアウト
            )

            execution_time = time.time() - start_time

            print(f"\\n実行時間: {execution_time:.2f}秒")
            print(f"リターンコード: {result.returncode}")

            if result.stdout:
                print("STDOUT:")
                print(result.stdout)

            if result.stderr:
                print("STDERR:")
                print(result.stderr)

            if result.returncode == 0 and output_path.exists():
                file_size = output_path.stat().st_size
                print(f"✅ Wav2Lip動画生成成功!")
                print(f"   出力: {output_path}")
                print(f"   ファイルサイズ: {file_size / 1024 / 1024:.2f}MB")
                return str(output_path)
            else:
                print(f"❌ Wav2Lip動画生成失敗 (code {result.returncode})")
                return None

        except subprocess.TimeoutExpired:
            print("❌ Wav2Lip実行タイムアウト (10分)")
            return None
        except Exception as e:
            print(f"❌ Wav2Lip実行中にエラー: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_integrated_pipeline(self, script_text, video_path, ref_audio_path, output_filename=None):
        """統合パイプライン実行"""
        print("=" * 60)
        print("🚀 SOVITS-Wav2Lip統合パイプライン開始")
        print("=" * 60)
        print(f"スクリプト: {script_text[:100]}...")
        print(f"動画: {video_path}")
        print(f"リファレンス音声: {ref_audio_path}")

        total_start_time = time.time()

        try:
            # フェーズ1: SOVITS音声クローン
            print("\\n" + "=" * 60)
            print("📍 フェーズ1: 音声クローン生成")
            print("=" * 60)

            cloned_audio_path = self.run_sovits_voice_clone(
                script_text,
                ref_audio_path
            )

            if not cloned_audio_path:
                print("❌ フェーズ1失敗: 音声クローン生成できませんでした")
                return None

            print(f"✅ フェーズ1完了: {cloned_audio_path}")

            # メモリクリーンアップのための待機
            print("\\n🔄 メモリクリーンアップ中...")
            time.sleep(2)

            # フェーズ2: Wav2Lip動画生成
            print("\\n" + "=" * 60)
            print("📍 フェーズ2: 口パク動画生成")
            print("=" * 60)

            final_video_path = self.run_wav2lip_with_cloned_voice(
                video_path,
                cloned_audio_path,
                output_filename
            )

            if not final_video_path:
                print("❌ フェーズ2失敗: 動画生成できませんでした")
                return None

            # 統計情報
            total_time = time.time() - total_start_time

            print("\\n" + "=" * 60)
            print("🎉 SOVITS-Wav2Lip統合処理完了!")
            print("=" * 60)
            print(f"総実行時間: {total_time:.2f}秒")
            print(f"出力動画: {final_video_path}")

            # 一時音声ファイルのクリーンアップ
            try:
                if cloned_audio_path and os.path.exists(cloned_audio_path):
                    if "temp_audio_" in cloned_audio_path:
                        os.remove(cloned_audio_path)
                        print(f"🗑️ 一時音声ファイル削除: {cloned_audio_path}")
            except Exception as e:
                print(f"警告: 一時ファイル削除エラー: {e}")

            return final_video_path

        except Exception as e:
            total_time = time.time() - total_start_time
            print("\\n" + "=" * 60)
            print("❌ 統合パイプラインエラー")
            print("=" * 60)
            print(f"エラー: {e}")
            print(f"実行時間: {total_time:.2f}秒")
            import traceback
            traceback.print_exc()
            return None

def main():
    """テスト用メイン関数"""
    try:
        # 統合システム初期化
        integration = SOVITSWav2LipIntegration()

        # テスト用パラメータ
        test_script = "こんにちは、これは音声クローンのテストです。"
        test_video = "sample_video.mp4"  # 実際のファイルパスに変更してください
        test_ref_audio = "models/gpt_sovits/e_01_08_extended.wav"

        print("テスト実行: SOVITS-Wav2Lip統合パイプライン")

        result = integration.run_integrated_pipeline(
            test_script,
            test_video,
            test_ref_audio,
            "test_integration_result.mp4"
        )

        if result:
            print(f"\\n🎉 統合テスト成功: {result}")
        else:
            print("\\n❌ 統合テスト失敗")

    except Exception as e:
        print(f"メイン関数エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()