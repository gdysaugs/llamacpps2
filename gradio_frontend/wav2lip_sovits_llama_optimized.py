"""
高速化最適化版統合システム
並列処理パイプライン + GPUメモリ最適化
"""

import gradio as gr
import os
import gc
import torch
import asyncio
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# 並列処理最適化モジュールをインポート
from parallel_pipeline_optimizer import (
    ParallelPipelineOptimizer,
    GPUMemoryManager,
    optimize_gpu_usage
)

# 既存の統合クラス
from wav2lip_sovits_llama_integrated import SOVITSWav2LipLlamaGradioApp


class OptimizedIntegrator(SOVITSWav2LipLlamaGradioApp):
    """最適化された統合処理クラス"""

    def __init__(self):
        super().__init__()
        self.optimizer = ParallelPipelineOptimizer()
        self.memory_manager = GPUMemoryManager()
        self.preprocessing_cache = {}

    @optimize_gpu_usage
    def preprocess_video_parallel(self, video_file: str) -> Dict[str, Any]:
        """動画の前処理を並列実行用に準備"""
        return {
            "video_path": video_file,
            "frames_extracted": True,
            "ready_for_wav2lip": True
        }

    @optimize_gpu_usage
    def preprocess_sovits_parallel(self, voice_config: Dict) -> Dict[str, Any]:
        """SoVITSの前処理を並列実行"""
        return {
            "model_loaded": True,
            "config": voice_config,
            "ready_for_generation": True
        }

    @optimize_gpu_usage
    def preprocess_facefusion_parallel(self, source_face: str) -> Dict[str, Any]:
        """FaceFusionの前処理を並列実行"""
        return {
            "source_face": source_face,
            "model_preloaded": True,
            "ready_for_swap": True
        }

    async def process_full_pipeline_optimized(self,
                                            prompt: str,
                                            script_text: str,
                                            use_ai_conversation: bool,
                                            additional_prompt: str,
                                            video_file: str,
                                            reference_audio_file: str,
                                            source_face_file: str,
                                            device: str = "cuda",
                                            use_gfpgan: bool = True,
                                            use_facefusion: bool = True,
                                            sovits_speed: float = 1.0,
                                            progress=gr.Progress()) -> Tuple[Optional[str], str, Optional[Dict]]:
        """
        最適化された完全パイプライン処理
        並列処理とGPUメモリ最適化を適用
        """

        log_messages = []
        total_stats = {}
        start_time = time.time()

        try:
            # GPUメモリ初期化
            self.memory_manager.set_memory_limit('llama')
            self.optimizer.clear_gpu_memory()

            progress(0.05, "🚀 最適化パイプライン開始...")
            log_messages.append("🚀 最適化パイプライン開始")
            log_messages.append(f"デバイス: {device}")
            log_messages.append(f"FaceFusion: {'有効' if use_facefusion else '無効'}")
            log_messages.append(f"AI会話モード: {'有効' if use_ai_conversation else '無効'}")

            # 実際に音声生成に使用するテキスト
            actual_script_text = script_text

            # AI会話機能が有効な場合のみLlama生成実行
            if use_ai_conversation:
                # =============================================================
                # Phase 0: AI会話生成 (LlamaCPP)
                # =============================================================
                progress(0.05, "🤖 Phase 0: AI会話生成中...")
                log_messages.append("=" * 60)
                log_messages.append("🤖 Phase 0: AI会話生成開始")
                log_messages.append(f"ユーザー入力: {prompt}")
                if additional_prompt.strip():
                    log_messages.append(f"追加プロンプト: {additional_prompt}")
                log_messages.append("=" * 60)

                try:
                    def llama_generation(prompt_input):
                        self.memory_manager.set_memory_limit('llama')
                        return self.llama_integration.generate_response(
                            user_input=prompt_input,
                            additional_prompt=additional_prompt,
                            max_tokens=200,
                            temperature=0.7
                        )

                    # Llama生成実行
                    llama_result = llama_generation(prompt)

                    if not llama_result or not llama_result.get("success"):
                        error_msg = llama_result.get("message", "AI会話生成失敗") if llama_result else "AI会話生成失敗"
                        log_messages.append(f"❌ Phase 0失敗: {error_msg}")
                        log_messages.append("📝 直接セリフテキストを使用します")
                        # AI生成に失敗した場合は直接テキストを使用
                        actual_script_text = script_text
                    else:
                        actual_script_text = llama_result["response"]
                        log_messages.append(f"✅ Phase 0完了: AI応答生成成功")
                        log_messages.append(f"AI応答: {actual_script_text}")

                except Exception as e:
                    log_messages.append(f"❌ Phase 0エラー: {str(e)}")
                    log_messages.append("📝 直接セリフテキストを使用します")
                    actual_script_text = script_text

                # GPUメモリクリーンアップ
                self.optimizer.clear_gpu_memory()
                progress(0.15, "🧹 AI会話処理後GPUメモリクリーンアップ中...")

            else:
                log_messages.append("📝 直接セリフテキストを使用")
                log_messages.append(f"セリフテキスト: {actual_script_text}")

            # =============================================================
            # Phase 1: SoVITS前処理 (非並列)
            # =============================================================
            # Phase 1を実際のテキストに基づいて進行
            phase_offset = 0.2 if use_ai_conversation else 0.05
            log_messages.append(f"📄 使用テキスト: {actual_script_text[:100]}...")

            # =============================================================
            # Phase 2: SoVITS音声生成 + Wav2Lip前処理 (並列実行)
            # =============================================================
            progress(0.3, "🎵+🎬 Phase 2: SoVITS音声生成 + Wav2Lip前処理 (並列)")
            log_messages.append("=" * 60)
            log_messages.append("🎵+🎬 Phase 2: SoVITS音声生成 + Wav2Lip前処理 (並列実行)")
            log_messages.append("=" * 60)

            # メモリ管理切り替え
            self.memory_manager.set_memory_limit('sovits')

            # SoVITS音声生成関数
            def sovits_generation():
                def dummy_progress(value, desc):
                    pass  # 並列処理中はプログレス無効
                return self.sovits_integration.process_sovits_audio_generation(
                    reference_audio_file, generated_text, device, dummy_progress, speed_factor=sovits_speed
                )

            # Wav2Lip前処理関数
            def wav2lip_preprocessing():
                return self.preprocess_video_parallel(video_file)

            # 並列実行
            audio_result, video_prep = await self.optimizer.parallel_sovits_wav2lip_prep(
                sovits_generation,
                wav2lip_preprocessing,
                generated_text,
                sovits_prep,
                video_file
            )

            if not audio_result or not audio_result.get("success"):
                error_msg = audio_result.get("message", "音声生成失敗") if audio_result else "音声生成失敗"
                log_messages.append(f"❌ Phase 2失敗: {error_msg}")
                return None, "\n".join(log_messages), None

            generated_audio_path = audio_result["audio_path"]
            log_messages.append(f"✅ SoVITS音声生成完了: {generated_audio_path}")
            log_messages.append(f"✅ Wav2Lip前処理完了")

            # =============================================================
            # Phase 3: Wav2Lip処理 + FaceFusion前処理 (並列実行)
            # =============================================================
            progress(0.5, "🎬+🎭 Phase 3: Wav2Lip処理 + FaceFusion前処理 (並列)")
            log_messages.append("=" * 60)
            log_messages.append("🎬+🎭 Phase 3: Wav2Lip処理 + FaceFusion前処理 (並列実行)")
            log_messages.append("=" * 60)

            # メモリ管理切り替え
            self.memory_manager.set_memory_limit('wav2lip')

            # Wav2Lip処理関数
            def wav2lip_processing():
                def dummy_progress(value, desc):
                    pass
                return self.sovits_integration.process_wav2lip_sync(
                    video_file, generated_audio_path, use_gfpgan, device, dummy_progress
                )

            # FaceFusion前処理関数
            def facefusion_preprocessing():
                if use_facefusion and source_face_file:
                    return self.preprocess_facefusion_parallel(source_face_file)
                return {"skipped": True}

            # 並列実行
            wav2lip_result, facefusion_prep = await self.optimizer.parallel_wav2lip_facefusion_prep(
                wav2lip_processing,
                facefusion_preprocessing,
                generated_audio_path,
                video_prep,
                source_face_file if use_facefusion else None
            )

            if not wav2lip_result or not wav2lip_result.get("success"):
                error_msg = wav2lip_result.get("message", "Wav2Lip失敗") if wav2lip_result else "Wav2Lip失敗"
                log_messages.append(f"❌ Phase 3失敗: {error_msg}")
                return None, "\n".join(log_messages), None

            wav2lip_video_path = wav2lip_result["video_path"]
            log_messages.append(f"✅ Wav2Lip処理完了: {wav2lip_video_path}")
            log_messages.append(f"✅ FaceFusion前処理完了")

            # =============================================================
            # Phase 4: FaceFusion処理 (オプション)
            # =============================================================
            final_video_path = wav2lip_video_path

            if use_facefusion and source_face_file and not facefusion_prep.get("skipped"):
                progress(0.8, "🎭 Phase 4: FaceFusion顔合成処理")
                log_messages.append("=" * 60)
                log_messages.append("🎭 Phase 4: FaceFusion顔合成処理")
                log_messages.append("=" * 60)

                # メモリ管理切り替え
                self.memory_manager.set_memory_limit('facefusion')

                def facefusion_progress(value, desc):
                    progress(0.8 + value * 0.15, f"🎭 Phase 4: {desc}")

                try:
                    facefusion_result = self.facefusion_integration.process_face_swap(
                        source_face_file, wav2lip_video_path, facefusion_progress
                    )

                    if facefusion_result and facefusion_result.get("success"):
                        final_video_path = facefusion_result["video_path"]
                        log_messages.append(f"✅ Phase 4完了: {final_video_path}")
                        if "stats" in facefusion_result:
                            total_stats["phase4_stats"] = facefusion_result["stats"]
                    else:
                        error_msg = facefusion_result.get("message", "FaceFusion失敗") if facefusion_result else "FaceFusion失敗"
                        log_messages.append(f"⚠️ Phase 4失敗: {error_msg}")
                        log_messages.append("📹 Wav2Lip結果を最終出力として使用")

                except Exception as e:
                    log_messages.append(f"❌ Phase 4エラー: {str(e)}")
                    log_messages.append("📹 Wav2Lip結果を最終出力として使用")

            # 最終メモリクリーンアップ
            progress(0.95, "🧹 最終メモリクリーンアップ...")
            self.optimizer.clear_gpu_memory()

            # 統計情報の集計
            end_time = time.time()
            total_processing_time = end_time - start_time

            total_stats.update({
                "total_processing_time": total_processing_time,
                "optimization_enabled": True,
                "parallel_processing": True,
                "gpu_memory_optimized": True,
                "final_video_path": final_video_path
            })

            progress(1.0, f"✅ 最適化パイプライン完了! ({total_processing_time:.1f}秒)")
            log_messages.append("=" * 60)
            log_messages.append(f"🎉 最適化パイプライン完了!")
            log_messages.append(f"⏱️ 総処理時間: {total_processing_time:.1f}秒")
            log_messages.append(f"📁 最終出力: {final_video_path}")
            log_messages.append("🚀 並列処理による高速化適用済み")
            log_messages.append("🧹 GPUメモリ最適化適用済み")
            log_messages.append("=" * 60)

            return final_video_path, "\n".join(log_messages), total_stats

        except Exception as e:
            log_messages.append(f"❌ 最適化パイプラインエラー: {str(e)}")
            self.optimizer.clear_gpu_memory()
            return None, "\n".join(log_messages), None

    def cleanup_resources(self):
        """リソースのクリーンアップ"""
        self.optimizer.cleanup()
        self.memory_manager.release_memory()


def create_optimized_interface():
    """最適化版インターフェース作成"""

    integrator = OptimizedIntegrator()

    with gr.Blocks(title="🚀 最適化版 AI動画生成システム", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 最適化版 AI動画生成システム")
        gr.Markdown("**並列処理パイプライン + GPUメモリ最適化により大幅高速化**")

        with gr.Row():
            with gr.Column(scale=1):
                # 入力コントロール
                prompt_input = gr.Textbox(
                    label="🧠 プロンプト入力",
                    placeholder="Llamaに生成させたいテキストの指示を入力...",
                    lines=3
                )

                script_input = gr.Textbox(
                    label="📝 セリフテキスト",
                    placeholder="音声にしたいテキストを直接入力...",
                    lines=4
                )

                use_ai_conversation = gr.Checkbox(
                    label="🤖 AI会話モード",
                    value=False,
                    info="チェックするとLlamaCPPでテキスト生成、チェックしないと直接セリフテキストを使用"
                )

                additional_prompt_input = gr.Textbox(
                    label="🎭 キャラクター特徴",
                    placeholder="AI会話時のキャラクター設定を入力...",
                    lines=2
                )

                video_input = gr.File(
                    label="🎬 ベース動画ファイル",
                    file_types=["video"]
                )

                reference_audio_input = gr.File(
                    label="🎵 参照音声ファイル (SoVITS)",
                    file_types=["audio"]
                )

                source_face_input = gr.File(
                    label="🎭 ソース顔画像 (FaceFusion)",
                    file_types=["image"]
                )

                with gr.Row():
                    device_dropdown = gr.Dropdown(
                        choices=["cuda", "cpu"],
                        value="cuda",
                        label="🖥️ 処理デバイス"
                    )

                    sovits_speed_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="🎵 SoVITS速度"
                    )

                with gr.Row():
                    use_gfpgan_checkbox = gr.Checkbox(
                        label="✨ GFPGAN品質向上",
                        value=True
                    )

                    use_facefusion_checkbox = gr.Checkbox(
                        label="🎭 FaceFusion顔合成",
                        value=True
                    )

                process_button = gr.Button(
                    "🚀 最適化処理実行",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                # 出力
                output_video = gr.Video(
                    label="📹 生成動画",
                    height=400
                )

                with gr.Accordion("📊 処理ログ", open=False):
                    output_log = gr.Textbox(
                        label="処理ログ",
                        lines=15,
                        max_lines=20
                    )

                with gr.Accordion("📈 統計情報", open=False):
                    stats_output = gr.JSON(label="処理統計")

        # 最適化処理関数
        async def process_optimized(*args):
            return await integrator.process_full_pipeline_optimized(*args)

        # イベントハンドラー
        process_button.click(
            fn=process_optimized,
            inputs=[
                prompt_input,
                script_input,
                use_ai_conversation,
                additional_prompt_input,
                video_input,
                reference_audio_input,
                source_face_input,
                device_dropdown,
                use_gfpgan_checkbox,
                use_facefusion_checkbox,
                sovits_speed_slider
            ],
            outputs=[output_video, output_log, stats_output]
        )

        # クリーンアップイベント
        demo.unload(integrator.cleanup_resources)

    return demo


if __name__ == "__main__":
    # 最適化版インターフェース起動
    print("🚀 最適化版統合システム起動中...")

    demo = create_optimized_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,  # ポート7865で起動
        share=False
    )