# 🎭 Wav2Lip-SoVITS-Llama Modal デプロイガイド

## 📋 概要
このガイドでは、Wav2Lip-SoVITS-LlamaをModal Cloudにデプロイする手順を説明します。

## 🚀 クイックスタート

### 1. Modal CLIのインストール
```bash
pip install modal
```

### 2. 認証設定
```bash
modal token set --token-id ak-ez2l4QZmyHdFEXSEgcmUQ4 --token-secret as-gnww62pMy15k1gcyuGTw7c
```

### 3. デプロイ実行
```bash
modal deploy modal_app.py
```

### 4. アクセス
デプロイ完了後、以下のURLでアクセス可能：
```
https://yourname--wav2lip-sovits-llama-run-gradio-app.modal.run
```

## 💰 コスト情報

### GPU使用料金（T4 GPU）
- **GPU**: T4（最安オプション）
- **メモリ**: 8GB RAM
- **料金**: 使用時間課金
- **Keep Warm**: 1台常時起動（コールドスタート回避）

### 推定コスト
- アイドル時: $0.000X/分
- 処理中: $0.00X/分
- 月額概算: 軽使用で$10-30程度

## 🔧 設定詳細

### GPU設定
```python
gpu="T4"  # 最安GPU
memory=8192  # 8GB RAM
timeout=1800  # 30分タイムアウト
```

### ボリューム設定
- `wav2lip-models`: モデルファイル保存
- `wav2lip-outputs`: 生成結果保存

## 📁 ファイル構成

```
wav2lip-project/
├── modal_app.py              # Modalデプロイスクリプト
├── MODAL_DEPLOY_README.md    # このファイル
└── gradio_frontend/
    └── wav2lip_sovits_llama_integrated.py  # 元のアプリ
```

## 🎬 使用方法

1. **動画ファイル**をアップロード（.mp4, .avi, .mov, .mkv, .webm）
2. **リファレンス音声**をアップロード（.mp3, .wav, .m4a, .aac, .flac）
3. **セリフテキスト**を入力（最大500文字）
4. **🚀 Wav2Lip生成開始**ボタンをクリック
5. 処理完了後、生成された動画をダウンロード

## 🛠️ トラブルシューティング

### よくある問題

#### デプロイエラー
```bash
# 再認証
modal token set --token-id ak-ez2l4QZmyHdFEXSEgcmUQ4 --token-secret as-gnww62pMy15k1gcyuGTw7c

# 再デプロイ
modal deploy modal_app.py
```

#### GPU メモリ不足
- **🧹 メモリクリーンアップ**ボタンを使用
- ファイルサイズを小さくする
- 処理時間を短縮する

#### タイムアウトエラー
- 30分以内に処理が完了するよう調整
- 大きなファイルは分割処理

## 📊 機能制限（Modal版）

### 現在実装済み
- ✅ 基本Wav2Lip処理
- ✅ Gradio Webインターフェース
- ✅ ファイルアップロード/ダウンロード
- ✅ T4 GPU自動スケーリング

### 今後実装予定
- 🔄 SoVITS音声合成統合
- 🔄 FaceFusion顔交換統合
- 🔄 高度な品質設定

## 🌐 Modal管理コマンド

### デプロイ状況確認
```bash
modal app list
```

### ログ確認
```bash
modal logs wav2lip-sovits-llama
```

### アプリ停止
```bash
modal app stop wav2lip-sovits-llama
```

## 📞 サポート

### 技術的な質問
- Modal公式ドキュメント: https://modal.com/docs
- GPU設定: T4が最適（コスト vs 性能）

### 注意事項
- **認証情報**は他人と共有しない
- **使用量**は定期的に確認
- **大容量ファイル**処理時は事前に容量確認

---

🎉 **Modal Cloudで高速GPU処理を体験してください！**