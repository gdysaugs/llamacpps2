#!/usr/bin/env python3
"""
GPT-SoVITS サブプロセス版 個別ファイル生成→ffmpeg結合スクリプト
長文を短文に分割して個別のPythonプロセスで音声生成し、ffmpegで結合
メモリ隔離・安定性・並列処理対応
"""

import os
import sys
import json
import re
import subprocess
import concurrent.futures
from pathlib import Path

def get_base_path():
    """実行ファイルまたはスクリプトの基準ディレクトリを取得"""
    if getattr(sys, 'frozen', False):
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).parent
    return base_path

def load_config():
    """設定ファイルを読み込み"""
    base_path = get_base_path()
    config_file = base_path / "tts_config.json"
    
    default_config = {
        "model_config_path": "gpt_sovits_full/GPT_SoVITS/configs/tts_infer.yaml",
        "ref_audio_path": "models/gpt_sovits/e_01_08_extended.wav",
        "prompt_text": "ああっ、気持ちいい。もっと、もっとして。",
        "output_dir": "output",
        "chunk_length": 50,
        "max_workers": 3,  # 並列プロセス数
        "tts_params": {
            "top_k": 5,
            "top_p": 1,
            "temperature": 2.0,
            "text_split_method": "cut0",
            "speed_factor": 1.2,
            "batch_size": 4,
            "parallel_infer": True,
            "streaming_mode": False,
            "seed": 42
        }
    }
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # デフォルト値とマージ
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config[key], dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in config[key]:
                        config[key][sub_key] = sub_value
    else:
        config = default_config
    
    return config

def split_text_by_length(text, max_length=50):
    """テキストを指定文字数で分割"""
    chunks = []
    current_chunk = ""
    
    sentences = re.split(r'([。！？])', text)
    merged_sentences = []
    
    # 区切り文字と文を結合
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            merged_sentences.append(sentences[i] + sentences[i + 1])
        else:
            merged_sentences.append(sentences[i])
    
    # 長さ制限で分割
    for sentence in merged_sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]

def generate_chunk_subprocess(chunk_data):
    """サブプロセスでチャンク音声生成"""
    chunk_text, chunk_file, chunk_num, base_path = chunk_data
    
    try:
        print(f"チャンク {chunk_num}: {chunk_text[:30]}{'...' if len(chunk_text) > 30 else ''} ({len(chunk_text)}文字)")
        
        # サブプロセス実行
        cmd = [
            sys.executable,  # 現在のPythonインタープリター
            str(base_path / "tts_single_process.py"),
            chunk_text,
            str(chunk_file)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2分でタイムアウト
            cwd=str(base_path)
        )
        
        if result.returncode == 0:
            if chunk_file.exists():
                file_size = chunk_file.stat().st_size / 1024 / 1024  # MB
                print(f"  → chunk_{chunk_num:03d}.wav ({file_size:.2f}MB)")
                return True, chunk_num, None
            else:
                return False, chunk_num, f"出力ファイルが作成されませんでした"
        else:
            error_msg = result.stderr.strip() if result.stderr else "不明なエラー"
            return False, chunk_num, f"プロセスエラー (code {result.returncode}): {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, chunk_num, "タイムアウト (120秒)"
    except Exception as e:
        return False, chunk_num, f"例外エラー: {e}"

def generate_separate_speech_subprocess(text, output_filename, config):
    """サブプロセス版 分割音声生成"""
    try:
        base_path = get_base_path()
        output_dir = base_path / config["output_dir"]
        output_dir.mkdir(exist_ok=True)
        
        # テキスト分割
        chunk_length = config.get("chunk_length", 50)
        text_chunks = split_text_by_length(text, chunk_length)
        
        if not text_chunks:
            print("❌ 分割できるテキストがありません")
            return None
            
        print(f"テキストを {len(text_chunks)} 個のチャンクに分割")
        for i, chunk in enumerate(text_chunks, 1):
            print(f"  チャンク{i}: {chunk}")
        
        # 一時ディレクトリ作成
        temp_dir = output_dir / f"temp_{output_filename.replace('.wav', '')}"
        temp_dir.mkdir(exist_ok=True)
        
        # チャンクファイル生成データ準備
        chunk_tasks = []
        chunk_files = []
        
        for i, chunk_text in enumerate(text_chunks, 1):
            chunk_file = temp_dir / f"chunk_{i:03d}.wav"
            chunk_files.append(chunk_file)
            chunk_tasks.append((chunk_text, chunk_file, i, base_path))
        
        try:
            # 並列処理でチャンク生成
            max_workers = min(config.get("max_workers", 3), len(text_chunks))
            failed_chunks = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {executor.submit(generate_chunk_subprocess, task): task for task in chunk_tasks}
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    success, chunk_num, error_msg = future.result()
                    if not success:
                        failed_chunks.append((chunk_num, error_msg))
                        print(f"❌ チャンク{chunk_num}生成失敗: {error_msg}")
            
            if failed_chunks:
                print(f"❌ {len(failed_chunks)}個のチャンクが失敗しました")
                return None
            
            # 存在するチャンクファイルを確認
            existing_chunks = [f for f in chunk_files if f.exists()]
            if not existing_chunks:
                print("❌ 生成されたチャンクファイルがありません")
                return None
            
            print(f"\\nffmpegで {len(existing_chunks)} 個のファイルを結合中...")
            
            # ffmpeg用ファイルリスト作成
            concat_list = temp_dir / "concat_list.txt"
            with open(concat_list, 'w', encoding='utf-8') as f:
                for chunk_file in sorted(existing_chunks):
                    f.write(f"file '{chunk_file.absolute()}'\\n")
            
            # ffmpegで結合
            output_path = output_dir / output_filename
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_list),
                '-c', 'copy',
                '-y',  # 上書き
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if output_path.exists():
                    file_size = output_path.stat().st_size / 1024 / 1024  # MB
                    
                    # 再生時間取得
                    try:
                        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(output_path)]
                        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                        if duration_result.returncode == 0:
                            duration = float(duration_result.stdout.strip())
                            print(f"✅ ffmpeg結合完了: {output_filename}")
                            print(f"   ファイルサイズ: {file_size:.2f}MB")
                            print(f"   再生時間: {duration:.2f}秒")
                            print(f"   総チャンク数: {len(existing_chunks)}")
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
                    if chunk_file.exists():
                        chunk_file.unlink()
                if concat_list.exists():
                    concat_list.unlink()
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
                print("一時ファイルを削除しました")
            except Exception as e:
                print(f"一時ファイル削除エラー: {e}")
                
    except Exception as e:
        print(f"❌ 分割音声生成エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """メイン関数"""
    try:
        print("GPT-SoVITS サブプロセス版 個別生成→ffmpeg結合システム")
        print("=" * 60)
        
        # ffmpegの確認
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            print("✅ ffmpeg確認OK")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ ffmpegが見つかりません。インストールしてPATHに追加してください。")
            return
        
        # 設定読み込み
        config = load_config()
        
        # 喘ぎ声中心の感情的なテスト（読点なしバージョン）
        panting_text = """あんっあんっはあっはあっ気持ちいいもっともっとしてああんっだめ感じちゃういくいっちゃうはあはあ息が息ができないあんあんあんやめてでも止めないでああっああっ限界もう限界よいくいくいっちゃうーはあっはあっまだ終わらないもっと感じたいあんっ気持ちよすぎるこんなの初めて体が震えて止まらないあんあんあん"""
        
        print(f"\\n喘ぎ声中心テスト開始（サブプロセス版・読点なし）... ({len(panting_text)}文字)")
        print(f"並列プロセス数: {config.get('max_workers', 3)}")
        
        result1 = generate_separate_speech_subprocess(panting_text, "panting_emotional_test_subprocess.wav", config)
        
        if result1:
            print(f"🎉 サブプロセス版テスト成功!")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()