# WSL再インストール後のセットアップ手順

## 2つの方法から選択

### 方法1: バックアップなし（推奨・簡単）
モデルファイルは自動ダウンロード（10-20分）で復元します。
**バックアップ不要！** → 「方法2」へ

### 方法2: バックアップあり（速い）
モデルファイルをバックアップしておけば復元が速い（数分）。

## 1. 前準備（WSL削除前に実行）- 方法2のみ

### バックアップが必要なファイル
Windowsの `C:\wav2lip_backup` にコピーしてください：

```bash
# バックアップスクリプトを実行（推奨）
./backup_before_wsl_reset.sh
```

または手動で：

```bash
# バックアップディレクトリを作成
mkdir -p /mnt/c/wav2lip_backup

# models フォルダをコピー（約8.3GB）
cp -r models /mnt/c/wav2lip_backup/

# 大きなモデルファイルをコピー（合計約300MB）
mkdir -p /mnt/c/wav2lip_backup/large_models
cp GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt /mnt/c/wav2lip_backup/large_models/
cp utils/detection_Resnet50_Final.pth /mnt/c/wav2lip_backup/large_models/
cp faceID/recognition.onnx /mnt/c/wav2lip_backup/large_models/
```

**注意**: venv（仮想環境）は約26GB以上あるため、バックアップせず再作成することを推奨します。

## 2. WSL再インストール後のセットアップ

### 2.1 リポジトリをクローン
```bash
cd ~
git clone https://github.com/gdysaugs/llamacpps2.git wav2lip-project
cd wav2lip-project
```

### 2.2 セットアップスクリプトを実行
```bash
chmod +x restore_from_backup.sh
./restore_from_backup.sh
```

または手動で以下を実行：

### 2.3 モデルファイルを復元
```bash
# models フォルダをコピー
cp -r /mnt/c/wav2lip_backup/models ./

# 大きなモデルファイルをコピー
mkdir -p GPT_SoVITS/pretrained_models/sv utils faceID
cp /mnt/c/wav2lip_backup/large_models/pretrained_eres2netv2w24s4ep4.ckpt GPT_SoVITS/pretrained_models/sv/
cp /mnt/c/wav2lip_backup/large_models/detection_Resnet50_Final.pth utils/
cp /mnt/c/wav2lip_backup/large_models/recognition.onnx faceID/
```

### 2.4 仮想環境を作成
```bash
# メインのvenv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate

# facefusion用
python3 -m venv facefusion_env
source facefusion_env/bin/activate
pip install -r requirements_facefusion.txt
deactivate

# gpt_sovits用
python3 -m venv gpt_sovits_env
source gpt_sovits_env/bin/activate
pip install -r requirements_gpt_sovits.txt
deactivate

# llama用
python3 -m venv llama_venv
source llama_venv/bin/activate
pip install llama-cpp-python
deactivate

# gradio_frontend用
cd gradio_frontend
python3 -m venv gradio_venv
source gradio_venv/bin/activate
pip install -r requirements.txt
deactivate
cd ..
```

### 2.5 動作確認
```bash
source venv/bin/activate
python wav2lip_inference.py --help
```

## 除外されたファイル一覧

### models/ （8.3GB）
- models/wav2lip/
- models/gpt_sovits/
- models/facefusion/
- models/retinaface/
- models/gfpgan/
- models/Berghof-NSFW-7B.i1-IQ4_XS.gguf（3.7GB - 除外推奨）

### 大きなモデルファイル（100MB超過）
- GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt (102.55MB)
- utils/detection_Resnet50_Final.pth (104.43MB)
- faceID/recognition.onnx (91.64MB)

### 仮想環境（26GB+ - 再作成推奨）
- venv/
- facefusion_env/
- gpt_sovits_env/
- llama_venv/
- gradio_frontend/gradio_venv/

## トラブルシューティング

### バックアップが見つからない場合
```bash
ls -lh /mnt/c/wav2lip_backup
```

### 権限エラーが出る場合
```bash
chmod +x *.sh
```

### pip インストールが遅い場合
```bash
pip install -r requirements.txt --no-cache-dir
```