#!/usr/bin/env python3
"""
GPT-SoVITS 個別ファイル生成→ffmpeg結合スクリプト
長文を短文に分割して個別wavファイル生成し、ffmpegで結合
"""

import os
import sys
import soundfile as sf
import json
import numpy as np
import re
import subprocess
from pathlib import Path

def get_base_path():
    """実行ファイルまたはスクリプトの基準ディレクトリを取得"""
    if getattr(sys, 'frozen', False):
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).parent
    return base_path

def setup_environment():
    """環境設定とパス設定"""
    base_path = get_base_path()
    gpt_sovits_dir = base_path / "gpt_sovits_full"
    if not gpt_sovits_dir.exists():
        raise FileNotFoundError(f"GPT-SoVITS directory not found: {gpt_sovits_dir}")
    
    os.chdir(str(gpt_sovits_dir))
    sys.path.insert(0, str(gpt_sovits_dir))
    sys.path.insert(0, str(gpt_sovits_dir / "GPT_SoVITS"))
    
    return base_path, gpt_sovits_dir

class SeparateTTS:
    def __init__(self):
        self.base_path, self.gpt_sovits_dir = setup_environment()
        self.config = self.load_config()
        self.tts_pipeline = None
        
    def load_config(self):
        """設定ファイルを読み込み"""
        config_file = self.base_path / "tts_config.json"
        
        default_config = {
            "model_config_path": "gpt_sovits_full/GPT_SoVITS/configs/tts_infer.yaml",
            "ref_audio_path": "models/gpt_sovits/e_01_08_extended.wav", 
            "prompt_text": "",
            "output_dir": "output",
            "chunk_length": 50,
            "tts_params": {
                "top_k": 5,
                "top_p": 1,
                "temperature": 1.5,
                "text_split_method": "cut0",
                "speed_factor": 1.2,
                "batch_size": 4,
                "parallel_infer": True,
                "streaming_mode": False,
                "ref_free": True,
                "seed": 42
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"警告: 設定ファイル読み込みエラー: {e}")
        
        return default_config
    
    def convert_to_wav(self, audio_path):
        """MP3や他の形式をWAVに自動変換"""
        import subprocess
        from pathlib import Path

        audio_path = Path(audio_path)

        # 既にWAVファイルの場合はそのまま返す
        if audio_path.suffix.lower() == '.wav':
            return str(audio_path)

        # WAV変換後のパス
        wav_path = audio_path.with_suffix('.wav')

        # 既に変換済みのWAVファイルが存在する場合
        if wav_path.exists():
            print(f"✅ 変換済みWAVファイルを使用: {wav_path.name}")
            return str(wav_path)

        print(f"🔄 {audio_path.suffix.upper()}をWAVに変換中: {audio_path.name} -> {wav_path.name}")

        try:
            # ffmpegで変換
            cmd = [
                'ffmpeg', '-i', str(audio_path),
                '-ar', '22050',  # サンプリングレート
                '-ac', '1',      # モノラル
                '-y',            # 上書き
                str(wav_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print(f"✅ WAV変換完了: {wav_path.name}")
                return str(wav_path)
            else:
                print(f"❌ WAV変換失敗: {result.stderr}")
                return str(audio_path)  # 元のファイルを返す

        except Exception as e:
            print(f"❌ WAV変換エラー: {e}")
            return str(audio_path)  # 元のファイルを返す

    def extend_short_audio(self, audio_path, min_duration=3.0):
        """短い音声ファイルを繰り返し延長して最小時間以上にする"""
        import soundfile as sf
        import numpy as np
        
        try:
            # 音声ファイル読み込み
            audio_data, sr = sf.read(audio_path)
            current_duration = len(audio_data) / sr
            
            if current_duration >= min_duration:
                return audio_path  # 既に十分な長さ
            
            print(f"リファレンス音声が短い ({current_duration:.2f}秒) -> {min_duration}秒に延長中...")
            
            # 必要な繰り返し回数を計算
            repeat_count = int(np.ceil(min_duration / current_duration))
            
            # 音声を繰り返し
            extended_audio = np.tile(audio_data, repeat_count)
            
            # 指定時間でカット
            target_samples = int(min_duration * sr)
            extended_audio = extended_audio[:target_samples]
            
            # 拡張版を保存
            extended_path = audio_path.replace('.wav', '_extended.wav')
            sf.write(extended_path, extended_audio, sr)
            
            final_duration = len(extended_audio) / sr
            print(f"✅ 音声延長完了: {final_duration:.2f}秒 ({repeat_count}回繰り返し)")
            
            return extended_path
            
        except Exception as e:
            print(f"❌ 音声延長エラー: {e}")
            return audio_path  # エラー時は元のパスを返す

    def init_tts(self):
        """TTS システムを初期化"""
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        
        # リファレンス音声の長さチェック・延長
        ref_audio_path_str = self.config["ref_audio_path"]
        # 絶対パスか相対パスかを判定
        if Path(ref_audio_path_str).is_absolute():
            ref_audio_path = Path(ref_audio_path_str)
        else:
            ref_audio_path = self.base_path / ref_audio_path_str

        if ref_audio_path.exists():
            extended_ref_path = self.extend_short_audio(str(ref_audio_path))
            # 延長されたパスで設定を更新（絶対パスのまま保持）
            if extended_ref_path != str(ref_audio_path):
                self.config["ref_audio_path"] = extended_ref_path
        
        model_config_path = self.config["model_config_path"]
        if model_config_path.startswith("gpt_sovits_full/"):
            config_path = self.base_path / model_config_path
        else:
            config_path = self.gpt_sovits_dir / model_config_path
            
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        tts_config = TTS_Config(str(config_path))
        self.tts_pipeline = TTS(tts_config)
        print("TTS システム初期化完了")
        return self.tts_pipeline
    
    def split_text(self, text, max_length=50):
        """テキストを適切な区切りで分割"""
        if len(text) <= max_length:
            return [text]

        chunks = []
        current_chunk = ""

        # 句点・読点・感嘆符・疑問符で分割
        sentences = re.split(r'([。、！？])', text)

        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]  # 文 + 句点
            else:
                sentence = sentences[i]

            # 空文字列をスキップ
            if not sentence.strip():
                continue

            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        # まだ長すぎる場合は強制分割
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                for i in range(0, len(chunk), max_length):
                    final_chunks.append(chunk[i:i + max_length])

        return final_chunks
    
    def generate_chunk_file(self, text_chunk, chunk_id, temp_dir):
        """1つのチャンクのwavファイルを生成"""
        # リファレンス音声を自動的にWAVに変換
        ref_audio_path_str = self.config["ref_audio_path"]
        # 絶対パスか相対パスかを判定
        if Path(ref_audio_path_str).is_absolute():
            raw_ref_path = Path(ref_audio_path_str)
        else:
            raw_ref_path = self.base_path / ref_audio_path_str

        ref_audio_path = self.convert_to_wav(raw_ref_path)

        # 短い音声を延長（すでに延長済みなら使用）
        if "_extended.wav" not in str(ref_audio_path):
            ref_audio_path = self.extend_short_audio(ref_audio_path)

        # 元のテキストを保存
        original_text = text_chunk

        # 10文字以下の短いテキストに固定音声を追加（無効化）
        padding_added = False
        # if len(text_chunk.strip()) <= 10:
        #     text_chunk = text_chunk + "おはようございます"
        #     padding_added = True
        #     print(f"短文パディング追加: '{original_text}' → '{text_chunk}'")

        # すべての句読点を読点に変換（READMEの指示通り）
        text_chunk = text_chunk.replace('。', '、')
        text_chunk = text_chunk.replace('！', '、')
        text_chunk = text_chunk.replace('？', '、')
        text_chunk = text_chunk.replace('.', '、')
        text_chunk = text_chunk.replace('!', '、')
        text_chunk = text_chunk.replace('?', '、')

        request = {
            "text": text_chunk,
            "text_lang": "ja",
            "ref_audio_path": str(ref_audio_path),
            "prompt_text": self.config.get("prompt_text", ""),
            "prompt_lang": "ja",
            "return_fragment": False,
            **self.config["tts_params"]
        }

        print(f"チャンク {chunk_id}: {original_text[:30]}... ({len(original_text)}文字)")
        print(f"  → 実際送信テキスト: '{text_chunk}'")
        
        tts_generator = self.tts_pipeline.run(request)
        
        audio_chunks = []
        sample_rate = None
        
        for sr, chunk in tts_generator:
            if sample_rate is None:
                sample_rate = sr
            audio_chunks.append(chunk)
        
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks, axis=0)

            # パディングが追加された場合、後半部分（おはようございます）をカット
            if padding_added:
                # 「おはようございます」の推定時間: 約2.5秒
                padding_duration = 2.5
                cut_samples = int(padding_duration * sample_rate)

                # 音声の長さをチェックしてカット
                if len(full_audio) > cut_samples:
                    full_audio = full_audio[:-cut_samples]
                    print(f"  → パディング音声カット: {padding_duration}秒削除")

            chunk_file = temp_dir / f"chunk_{chunk_id:03d}.wav"
            sf.write(str(chunk_file), full_audio, sample_rate)
            print(f"  → {chunk_file.name}")
            return str(chunk_file)
        else:
            print(f"❌ チャンク {chunk_id} 生成失敗")
            return None
    
    def generate_separate_speech(self, text, output_filename=None):
        """個別ファイル生成→ffmpeg結合で音声生成"""
        if self.tts_pipeline is None:
            raise RuntimeError("TTS システムが初期化されていません")
        
        # テキスト分割
        chunk_length = self.config.get("chunk_length", 50)
        text_chunks = self.split_text(text, chunk_length)
        
        print(f"テキストを {len(text_chunks)} 個のチャンクに分割")
        for i, chunk in enumerate(text_chunks, 1):
            print(f"  チャンク{i}: {chunk}")
        
        # 一時ディレクトリ作成
        output_dir = self.base_path / self.config["output_dir"]
        output_dir.mkdir(exist_ok=True)
        temp_dir = output_dir / "temp_chunks"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # 各チャンクのwavファイルを生成
            chunk_files = []
            for i, text_chunk in enumerate(text_chunks, 1):
                # 最後のチャンクのみ文末句読点を削除
                if i == len(text_chunks):
                    text_chunk = text_chunk.rstrip('、。！？.!?')

                chunk_file = self.generate_chunk_file(text_chunk, i, temp_dir)
                if chunk_file:
                    chunk_files.append(chunk_file)
            
            if not chunk_files:
                print("❌ 全チャンク生成失敗")
                return None
            
            # 出力ファイル名設定
            if output_filename is None:
                import time
                timestamp = int(time.time())
                output_filename = f"separate_{timestamp}.wav"
            
            output_path = output_dir / output_filename
            
            # ffmpegで結合
            print(f"\\nffmpegで {len(chunk_files)} 個のファイルを結合中...")
            concat_list = temp_dir / "concat_list.txt"
            
            # concat用ファイルリスト作成（順序を確実にする）
            with open(concat_list, 'w', encoding='utf-8') as f:
                for chunk_file in sorted(chunk_files, key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in str(x) else 0):
                    f.write(f"file '{os.path.basename(chunk_file)}'\n")
            
            # ffmpeg実行
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # 上書き許可
                '-f', 'concat',  # concat形式
                '-safe', '0',    # パス制限無効
                '-i', str(concat_list),  # 入力リスト
                '-c', 'copy',    # コーデックコピー（再エンコードなし）
                str(output_path)  # 出力ファイル
            ]
            
            result = subprocess.run(
                ffmpeg_cmd, 
                cwd=str(temp_dir),  # 作業ディレクトリを一時ディレクトリに
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                # ファイルサイズと再生時間を取得
                if output_path.exists():
                    file_size = output_path.stat().st_size / 1024 / 1024  # MB
                    
                    # ffprobeで再生時間を取得
                    probe_cmd = [
                        'ffprobe', '-v', 'quiet',
                        '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        str(output_path)
                    ]
                    
                    try:
                        duration_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                        if duration_result.returncode == 0:
                            duration = float(duration_result.stdout.strip())
                            print(f"✅ ffmpeg結合完了: {output_filename}")
                            print(f"   ファイルサイズ: {file_size:.2f}MB")
                            print(f"   再生時間: {duration:.2f}秒")
                            print(f"   総チャンク数: {len(text_chunks)}")
                        else:
                            print(f"✅ ffmpeg結合完了: {output_filename} (サイズ: {file_size:.2f}MB)")
                    except:
                        print(f"✅ ffmpeg結合完了: {output_filename} (サイズ: {file_size:.2f}MB)")
                    
                    return str(output_path)
                else:
                    print("❌ 出力ファイルが作成されませんでした")
                    return None
            else:
                print(f"❌ ffmpeg結合エラー: {result.stderr}")
                return None
        
        finally:
            # デバッグ用：一時ファイル削除を無効化
            print("デバッグモード: 一時ファイルを保持します")
            print(f"一時ディレクトリ: {temp_dir}")
            if chunk_files:
                print("生成されたチャンクファイル:")
                for i, chunk_file in enumerate(chunk_files, 1):
                    print(f"  チャンク{i}: {chunk_file}")
            # try:
            #     for chunk_file in chunk_files:
            #         if os.path.exists(chunk_file):
            #             os.remove(chunk_file)
            #     if concat_list.exists():
            #         concat_list.unlink()
            #     if temp_dir.exists() and not any(temp_dir.iterdir()):
            #         temp_dir.rmdir()
            #     print("一時ファイルを削除しました")
            # except Exception as e:
            #     print(f"一時ファイル削除エラー: {e}")

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPT-SoVITS 長文対応音声生成')
    parser.add_argument('text', nargs='?', help='生成するテキスト')
    parser.add_argument('output', nargs='?', help='出力ファイル名')
    args = parser.parse_args()
    
    try:
        print("GPT-SoVITS 個別生成→ffmpeg結合システム")
        print("=" * 50)
        
        # ffmpegの確認
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            print("✅ ffmpeg確認OK")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ ffmpegが見つかりません。インストールしてPATHに追加してください。")
            return
        
        separate_tts = SeparateTTS()
        separate_tts.init_tts()
        
        # コマンドライン引数がある場合はそれを使用
        if args.text and args.output:
            print(f"\\nコマンドライン引数で音声生成... ({len(args.text)}文字)")
            result = separate_tts.generate_separate_speech(args.text, args.output)
            if result:
                print(f"🎉 音声生成成功: {args.output}")
            else:
                sys.exit(1)
        else:
            # 従来のテストコード
            panting_text = """あんっあんっはあっはあっ気持ちいいもっともっとしてああんっだめ感じちゃういくいっちゃうはあはあ息が息ができないあんあんあんやめてでも止めないでああっああっ限界もう限界よいくいくいっちゃうーはあっはあっまだ終わらないもっと感じたいあんっ気持ちよすぎるこんなの初めて体が震えて止まらないあんあんあん"""
            
            print(f"\\n喘ぎ声中心テスト開始（読点なし）... ({len(panting_text)}文字)")
            result1 = separate_tts.generate_separate_speech(panting_text, "panting_emotional_test_no_comma.wav")
            
            if result1:
                print(f"🎉 喘ぎ声中心テスト成功!")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()