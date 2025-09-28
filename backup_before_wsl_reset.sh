#!/bin/bash

echo "========================================="
echo "WSL削除前のバックアップスクリプト"
echo "========================================="
echo ""

BACKUP_DIR="/mnt/c/wav2lip_backup"

# バックアップディレクトリの作成
echo "バックアップディレクトリを作成中: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR/large_models"
echo "✓ ディレクトリを作成しました"
echo ""

# 1. modelsフォルダのバックアップ
echo "========================================="
echo "1. models/ フォルダをバックアップ中..."
echo "========================================="
echo "これには数分かかります（約8.3GB）..."

if [ -d "models" ]; then
    cp -r models "$BACKUP_DIR/"
    echo "✓ models/ フォルダをバックアップしました"
else
    echo "⚠ 警告: models/ フォルダが見つかりません"
fi

echo ""

# 2. 大きなモデルファイルのバックアップ
echo "========================================="
echo "2. 大きなモデルファイルをバックアップ中..."
echo "========================================="

if [ -f "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt" ]; then
    cp "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt" "$BACKUP_DIR/large_models/"
    echo "✓ pretrained_eres2netv2w24s4ep4.ckpt (102.55MB)"
else
    echo "⚠ 警告: pretrained_eres2netv2w24s4ep4.ckpt が見つかりません"
fi

if [ -f "utils/detection_Resnet50_Final.pth" ]; then
    cp "utils/detection_Resnet50_Final.pth" "$BACKUP_DIR/large_models/"
    echo "✓ detection_Resnet50_Final.pth (104.43MB)"
else
    echo "⚠ 警告: detection_Resnet50_Final.pth が見つかりません"
fi

if [ -f "faceID/recognition.onnx" ]; then
    cp "faceID/recognition.onnx" "$BACKUP_DIR/large_models/"
    echo "✓ recognition.onnx (91.64MB)"
else
    echo "⚠ 警告: recognition.onnx が見つかりません"
fi

echo ""

# 3. バックアップ内容の確認
echo "========================================="
echo "3. バックアップ内容を確認中..."
echo "========================================="

echo ""
echo "バックアップされたファイル："
du -sh "$BACKUP_DIR"/*
echo ""

echo "バックアップディレクトリの詳細："
echo "  場所: $BACKUP_DIR"
echo "  Windowsからアクセス: C:\\wav2lip_backup"
echo ""

# 4. バックアップの完了メッセージ
echo "========================================="
echo "✓ バックアップ完了！"
echo "========================================="
echo ""
echo "次のステップ："
echo "1. WSLをアンインストール"
echo "2. WSLを再インストール"
echo "3. リポジトリをクローン："
echo "   git clone https://github.com/gdysaugs/llamacpps2.git wav2lip-project"
echo "4. 復元スクリプトを実行："
echo "   cd wav2lip-project"
echo "   chmod +x restore_from_backup.sh"
echo "   ./restore_from_backup.sh"
echo ""
echo "⚠ 注意: バックアップが完了したことを確認してから WSL を削除してください"
echo "        C:\\wav2lip_backup の内容を確認してください"
echo ""