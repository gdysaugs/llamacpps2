#!/usr/bin/env python3
"""
Audio Silence Reducer
1秒以上の無音区間を自動的に1秒に短縮する
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple


class AudioSilenceReducer:
    def __init__(self,
                 silence_threshold: float = 0.01,  # 無音判定の閾値
                 max_silence_duration: float = 1.0,  # 最大無音時間（秒）
                 min_silence_duration: float = 0.1,  # 最小無音時間（秒）
                 sample_rate: int = 22050):
        """
        Args:
            silence_threshold: 無音判定の閾値（0-1）
            max_silence_duration: 最大無音時間（秒）
            min_silence_duration: 最小無音時間（秒）
            sample_rate: サンプルレート
        """
        self.silence_threshold = silence_threshold
        self.max_silence_duration = max_silence_duration
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate

    def detect_silence_regions(self, audio: np.ndarray,
                             hop_length: int = 512) -> list:
        """
        無音区間を検出

        Args:
            audio: 音声データ
            hop_length: フレーム間隔

        Returns:
            List[Tuple[int, int]]: 無音区間のリスト（開始サンプル, 終了サンプル）
        """
        # RMS エネルギーを計算
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

        # フレームを秒に変換するための係数
        frame_to_time = hop_length / self.sample_rate

        # 無音判定
        silence_frames = rms < self.silence_threshold

        # 連続する無音フレームをグループ化
        silence_regions = []
        start_frame = None

        for i, is_silence in enumerate(silence_frames):
            if is_silence and start_frame is None:
                start_frame = i
            elif not is_silence and start_frame is not None:
                # 無音区間終了
                duration = (i - start_frame) * frame_to_time
                if duration >= self.min_silence_duration:
                    start_sample = start_frame * hop_length
                    end_sample = i * hop_length
                    silence_regions.append((start_sample, end_sample))
                start_frame = None

        # 最後が無音で終わる場合
        if start_frame is not None:
            duration = (len(silence_frames) - start_frame) * frame_to_time
            if duration >= self.min_silence_duration:
                start_sample = start_frame * hop_length
                end_sample = len(audio)
                silence_regions.append((start_sample, end_sample))

        return silence_regions

    def reduce_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        1秒以上の無音区間を1秒に短縮

        Args:
            audio: 入力音声データ

        Returns:
            処理済み音声データ
        """
        silence_regions = self.detect_silence_regions(audio)

        if not silence_regions:
            return audio

        # 処理後の音声を構築
        processed_segments = []
        last_end = 0

        for start_sample, end_sample in silence_regions:
            # 無音区間前の音声を追加
            if start_sample > last_end:
                processed_segments.append(audio[last_end:start_sample])

            # 無音区間の長さを計算
            silence_duration = (end_sample - start_sample) / self.sample_rate

            if silence_duration > self.max_silence_duration:
                # 1秒に短縮
                new_silence_samples = int(self.max_silence_duration * self.sample_rate)
                reduced_silence = np.zeros(new_silence_samples, dtype=audio.dtype)
                processed_segments.append(reduced_silence)
                print(f"🔇 無音区間短縮: {silence_duration:.2f}秒 → {self.max_silence_duration:.2f}秒")
            else:
                # そのまま保持
                processed_segments.append(audio[start_sample:end_sample])

            last_end = end_sample

        # 最後の音声部分を追加
        if last_end < len(audio):
            processed_segments.append(audio[last_end:])

        if processed_segments:
            return np.concatenate(processed_segments)
        else:
            return audio

    def process_audio_file(self, input_path: str, output_path: str) -> Tuple[bool, str]:
        """
        音声ファイルを処理

        Args:
            input_path: 入力ファイルパス
            output_path: 出力ファイルパス

        Returns:
            Tuple[bool, str]: (成功フラグ, メッセージ)
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)

            if not input_path.exists():
                return False, f"入力ファイルが見つかりません: {input_path}"

            print(f"🎵 音声無音削減処理開始: {input_path}")

            # 音声ファイルを読み込み
            audio, sr = librosa.load(str(input_path), sr=self.sample_rate)
            original_duration = len(audio) / sr

            print(f"📊 元の音声時間: {original_duration:.2f}秒")

            # 無音削減処理
            processed_audio = self.reduce_silence(audio)
            processed_duration = len(processed_audio) / sr

            print(f"📊 処理後音声時間: {processed_duration:.2f}秒")
            time_saved = original_duration - processed_duration

            if time_saved > 0.1:  # 0.1秒以上短縮された場合
                print(f"⏰ 短縮時間: {time_saved:.2f}秒")

            # 出力ディレクトリを作成
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 処理済み音声を保存
            sf.write(str(output_path), processed_audio, sr)

            return True, f"無音削減完了: {time_saved:.2f}秒短縮"

        except Exception as e:
            error_msg = f"音声無音削減エラー: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg

    def get_audio_info(self, audio_path: str) -> dict:
        """
        音声ファイルの情報を取得

        Args:
            audio_path: 音声ファイルパス

        Returns:
            音声情報の辞書
        """
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr

            # 無音区間を検出
            silence_regions = self.detect_silence_regions(audio)
            silence_count = len(silence_regions)

            total_silence_duration = 0
            long_silence_count = 0

            for start, end in silence_regions:
                silence_duration = (end - start) / sr
                total_silence_duration += silence_duration
                if silence_duration > self.max_silence_duration:
                    long_silence_count += 1

            return {
                "duration": duration,
                "sample_rate": sr,
                "silence_regions": silence_count,
                "total_silence_duration": total_silence_duration,
                "long_silence_count": long_silence_count,
                "potential_time_saving": max(0, total_silence_duration - (silence_count * self.max_silence_duration))
            }

        except Exception as e:
            return {"error": str(e)}


def reduce_audio_silence(input_path: str,
                        output_path: Optional[str] = None,
                        max_silence_duration: float = 1.0) -> Tuple[bool, str, str]:
    """
    便利関数：音声ファイルの無音削減

    Args:
        input_path: 入力ファイルパス
        output_path: 出力ファイルパス（Noneの場合は入力ファイルを上書き）
        max_silence_duration: 最大無音時間（秒）

    Returns:
        Tuple[bool, str, str]: (成功フラグ, メッセージ, 出力ファイルパス)
    """
    if output_path is None:
        # 一時ファイルに出力して、後で元ファイルに置き換え
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_reduced{input_path_obj.suffix}")

    reducer = AudioSilenceReducer(max_silence_duration=max_silence_duration)
    success, message = reducer.process_audio_file(input_path, output_path)

    if success and output_path != input_path:
        # 元ファイルを置き換え
        try:
            import shutil
            shutil.move(output_path, input_path)
            output_path = input_path
        except Exception as e:
            return False, f"ファイル置き換えエラー: {e}", output_path

    return success, message, output_path


if __name__ == "__main__":
    # テスト用
    import sys

    if len(sys.argv) < 2:
        print("使用法: python audio_silence_reducer.py <input_audio_file> [output_audio_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success, message, output_path = reduce_audio_silence(input_file, output_file)

    if success:
        print(f"✅ {message}")
        print(f"📁 出力ファイル: {output_path}")
    else:
        print(f"❌ {message}")
        sys.exit(1)