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
            "prompt_text": "ああっ、気持ちいい。もっと、もっとして。",
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
    
    def init_tts(self):
        """TTS システムを初期化"""
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        
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
        
        # 句点でのみ分割（READMEの推奨通り）
        sentences = re.split(r'(。)', text)
        
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
    
    def generate_chunk_subprocess(self, text_chunk, chunk_id, temp_dir):
        """サブプロセスでチャンク生成"""
        import subprocess
        try:
            chunk_file = temp_dir / f"chunk_{chunk_id:03d}.wav"
            
            # 一時的な単発スクリプトを作成
            single_script = temp_dir / f"single_{chunk_id:03d}.py"
            self.create_single_tts_script(single_script, text_chunk, str(chunk_file))
            
            print(f"チャンク {chunk_id}: {text_chunk[:30]}... ({len(text_chunk)}文字)")
            
            # サブプロセス実行
            cmd = [sys.executable, str(single_script)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.gpt_sovits_dir),
                env=dict(os.environ, PYTHONPATH=str(self.gpt_sovits_dir))
            )
            
            # 一時スクリプト削除
            if single_script.exists():
                single_script.unlink()
            
            if result.returncode == 0 and chunk_file.exists():
                print(f"  → chunk_{chunk_id:03d}.wav")
                return chunk_file
            else:
                print(f"❌ チャンク {chunk_id} サブプロセス失敗: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ チャンク {chunk_id} サブプロセスエラー: {e}")
            return None
    
    def create_single_tts_script(self, script_path, text, output_file):
        """単発TTS実行スクリプトを動的生成"""
        script_content = f'''#!/usr/bin/env python3
import os
import sys
import soundfile as sf
import json
from pathlib import Path

# 環境設定
gpt_sovits_dir = Path("{str(self.gpt_sovits_dir)}")
sys.path.insert(0, str(gpt_sovits_dir))
sys.path.insert(0, str(gpt_sovits_dir / "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

try:
    # TTS初期化
    model_config_path = Path("{str(self.base_path / self.config['model_config_path'])}")
    tts_config = TTS_Config(str(model_config_path))
    tts_pipeline = TTS(tts_config)
    
    # 音声生成
    ref_audio_path = "{str(self.base_path / self.config['ref_audio_path'])}"
    text_chunk = """{text}"""
    
    # 句読点変換
    text_chunk = text_chunk.replace('。', '、')
    text_chunk = text_chunk.replace('！', '、')
    text_chunk = text_chunk.replace('？', '、')
    text_chunk = text_chunk.replace('.', '、')
    text_chunk = text_chunk.replace('!', '、')
    text_chunk = text_chunk.replace('?', '、')
    
    request = {{
        "text": text_chunk,
        "text_lang": "ja",
        "ref_audio_path": ref_audio_path,
        "prompt_text": "{self.config['prompt_text']}",
        "prompt_lang": "ja",
        "return_fragment": False,
        {repr(self.config['tts_params'])[1:-1]}
    }}
    
    sr, audio_data = tts_pipeline.run(request)
    
    if audio_data is not None:
        sf.write("{output_file}", audio_data, sr)
        sys.exit(0)
    else:
        sys.exit(1)
        
except Exception as e:
    print(f"Error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
    
    def generate_chunk_file(self, text_chunk, chunk_id, temp_dir):
        """1つのチャンクのwavファイルを生成"""
        ref_audio_path = self.base_path / self.config["ref_audio_path"]
        
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
            "prompt_text": self.config["prompt_text"],
            "prompt_lang": "ja",
            "return_fragment": False,
            **self.config["tts_params"]
        }
        
        print(f"チャンク {chunk_id}: {text_chunk[:30]}... ({len(text_chunk)}文字)")
        
        tts_generator = self.tts_pipeline.run(request)
        
        audio_chunks = []
        sample_rate = None
        
        for sr, chunk in tts_generator:
            if sample_rate is None:
                sample_rate = sr
            audio_chunks.append(chunk)
        
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks, axis=0)
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
            # 各チャンクのwavファイルを生成（サブプロセス版）
            chunk_files = []
            import concurrent.futures
            
            # 並列処理でサブプロセス実行
            max_workers = 3
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {
                    executor.submit(self.generate_chunk_subprocess, text_chunk, i, temp_dir): i 
                    for i, text_chunk in enumerate(text_chunks, 1)
                }
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    try:
                        chunk_file = future.result()
                        if chunk_file:
                            chunk_files.append(chunk_file)
                    except Exception as e:
                        print(f"❌ チャンク {chunk_id} 並列処理エラー: {e}")
            
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
            
            # concat用ファイルリスト作成
            with open(concat_list, 'w', encoding='utf-8') as f:
                for chunk_file in chunk_files:
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
            # 一時ファイル削除
            try:
                for chunk_file in chunk_files:
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                if concat_list.exists():
                    concat_list.unlink()
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
                print("一時ファイルを削除しました")
            except Exception as e:
                print(f"一時ファイル削除エラー: {e}")

def main():
    """メイン関数"""
    try:
        print("GPT-SoVITS サブプロセス版 個別生成→ffmpeg結合システム")
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
        
        # 喘ぎ声中心の感情的なテスト（読点なしバージョン）
        panting_text = """あんっあんっはあっはあっ気持ちいいもっともっとしてああんっだめ感じちゃういくいっちゃうはあはあ息が息ができないあんあんあんやめてでも止めないでああっああっ限界もう限界よいくいくいっちゃうーはあっはあっまだ終わらないもっと感じたいあんっ気持ちよすぎるこんなの初めて体が震えて止まらないあんあんあん"""
        
        print(f"\\n喘ぎ声中心テスト開始（サブプロセス版・読点なし）... ({len(panting_text)}文字)")
        result1 = separate_tts.generate_separate_speech(panting_text, "panting_emotional_test_subprocess.wav")
        
        if result1:
            print(f"🎉 喘ぎ声中心テスト成功!")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()