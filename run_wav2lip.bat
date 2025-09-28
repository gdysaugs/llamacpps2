@echo off
REM SoVITS-Wav2Lip-LlamaCPP統合システム ポータブル版起動スクリプト
REM Windows用バッチファイル

title SoVITS-Wav2Lip-LlamaCPP統合システム

echo.
echo ======================================================
echo   SoVITS-Wav2Lip-LlamaCPP統合システム ポータブル版
echo ======================================================
echo.

REM 現在のディレクトリをアプリケーションルートに設定
cd /d "%~dp0"

REM 環境変数設定
set PYTHONPATH=%cd%;%cd%\gradio_frontend;%cd%\python\Lib\site-packages
set PATH=%cd%\python;%cd%\python\Scripts;%PATH%

REM CUDA環境確認
echo 🔍 CUDA環境確認中...
python\python.exe -c "import torch; print('CUDA利用可能:', torch.cuda.is_available()); print('GPU数:', torch.cuda.device_count())" 2>nul
if errorlevel 1 (
    echo ⚠️  PyTorchまたはCUDAが利用できません。CPUモードで動作します。
) else (
    echo ✅ CUDA環境確認完了
)
echo.

REM 必要なディレクトリ作成
if not exist "output" mkdir output
if not exist "temp" mkdir temp
if not exist "models" mkdir models

echo 📁 ディレクトリ構造確認完了
echo.

REM モデルファイル確認
echo 🔍 モデルファイル確認中...
if exist "models\wav2lip_gan.pth" (
    echo ✅ Wav2Lipモデル: 確認済み
) else (
    echo ❌ Wav2Lipモデル: models\wav2lip_gan.pth が見つかりません
    echo    初回起動時は自動ダウンロードが実行されます
)

if exist "models\gpt_sovits" (
    echo ✅ GPT-SoVITSモデル: 確認済み
) else (
    echo ❌ GPT-SoVITSモデル: models\gpt_sovits が見つかりません
)
echo.

REM Pythonスクリプト実行
echo 🚀 アプリケーション起動中...
echo    Web UI: http://localhost:7866
echo    終了するには Ctrl+C を押してください
echo.

python\python.exe gradio_frontend\wav2lip_sovits_llama_integrated_portable.py

echo.
echo アプリケーションが終了しました。
pause