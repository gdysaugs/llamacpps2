#!/usr/bin/env python3
"""
Wav2Lip inference with RetinaFace GPU detection
Based on https://github.com/instant-high/wav2lip-onnx-HQ official implementation
"""

import os
import sys
import cv2
import numpy as np
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import onnxruntime as ort
import gc
import os

# Suppress ONNX runtime logs
ort.set_default_logger_severity(4)  # Only show fatal errors

# Set environment variables to avoid CUDA errors
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import audio processing
import audio
from hparams import hparams as hp

# Import RetinaFace detection
from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head_256

class Wav2LipRetinaGPU:
    def __init__(self, checkpoint_path, device='cuda', use_gfpgan=False):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.img_size = 96
        self.mel_step_size = 16
        self.pads = 0  # face crop padding
        self.face_mode = 0  # 0=portrait, 1=square
        self.use_gfpgan = use_gfpgan

        # Load models
        self.load_wav2lip_model()
        self.load_face_detector()
        if self.use_gfpgan:
            self.load_gfpgan()

        print(f"✓ Models loaded successfully on {device}")

    def load_wav2lip_model(self):
        """Load Wav2Lip ONNX model with safe GPU support"""
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Force CUDA provider setup
        providers = ["CPUExecutionProvider"]
        if self.device == 'cuda':
            try:
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    # CUDA 12.6 compatible options (simplified)
                    cuda_options = {
                        'device_id': 0,
                        'cudnn_conv_algo_search': 'DEFAULT',
                    }
                    providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
                    print("✓ Using CUDA for Wav2Lip with forced GPU memory")
                else:
                    print("Warning: CUDA not available for Wav2Lip, using CPU")
            except Exception as e:
                print(f"Warning: CUDA setup failed for Wav2Lip: {e}")

        self.model = ort.InferenceSession(
            self.checkpoint_path,
            sess_options=session_options,
            providers=providers
        )

        # Check actual providers being used
        actual_providers = self.model.get_providers()
        print(f"✓ Wav2Lip model loaded: {self.checkpoint_path}")
        print(f"✓ Wav2Lip actual providers: {actual_providers}")

    def load_face_detector(self):
        """Load RetinaFace detector with safe GPU support"""
        # Safe CUDA provider setup
        providers = ["CPUExecutionProvider"]
        if self.device == 'cuda':
            try:
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    print("✓ Using CUDA for RetinaFace")
                else:
                    print("Warning: CUDA not available for RetinaFace, using CPU")
            except Exception as e:
                print(f"Warning: CUDA setup failed for RetinaFace: {e}")

        self.detector = RetinaFace(
            "utils/scrfd_2.5g_bnkps.onnx",
            provider=providers,
            session_options=None
        )
        print("✓ RetinaFace detector loaded")

    def load_gfpgan(self):
        """Load GFPGAN face enhancer using official implementation"""
        try:
            gfpgan_path = "models/gfpgan/GFPGANv1.4.onnx"
            if not os.path.exists(gfpgan_path):
                print(f"Warning: GFPGAN model not found at {gfpgan_path}")
                self.use_gfpgan = False
                return

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Force CUDA provider setup for GFPGAN
            providers = ["CPUExecutionProvider"]
            if self.device == 'cuda':
                try:
                    available_providers = ort.get_available_providers()
                    if 'CUDAExecutionProvider' in available_providers:
                        # CUDA 12.6 compatible options for GFPGAN (simplified)
                        cuda_options = {
                            'device_id': 0,
                            'cudnn_conv_algo_search': 'DEFAULT',
                        }
                        providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
                        print("✓ Using CUDA for GFPGAN with forced GPU memory")
                    else:
                        print("Warning: CUDA not available for GFPGAN, using CPU")
                except Exception as e:
                    print(f"Warning: CUDA setup failed for GFPGAN: {e}")

            self.gfpgan_model = ort.InferenceSession(
                gfpgan_path,
                sess_options=session_options,
                providers=providers
            )

            # Check actual providers being used
            actual_providers = self.gfpgan_model.get_providers()
            print(f"✓ GFPGAN actual providers: {actual_providers}")

            # Get resolution from model input shape
            self.gfpgan_resolution = self.gfpgan_model.get_inputs()[0].shape[-2:]
            print(f"✓ GFPGAN model loaded: {gfpgan_path} (resolution: {self.gfpgan_resolution})")
        except Exception as e:
            print(f"Warning: Failed to load GFPGAN: {e}")
            self.use_gfpgan = False

    def enhance_face_gfpgan(self, face_img):
        """Apply GFPGAN face enhancement using official preprocessing"""
        try:
            # Check if GPU is being used (only once)
            if not hasattr(self, '_gfpgan_providers_checked'):
                providers = self.gfpgan_model.get_providers()
                print(f"GFPGAN providers: {providers}")
                self._gfpgan_providers_checked = True

            original_size = face_img.shape[:2]

            # Official preprocess: resize to model resolution
            img = cv2.resize(face_img, self.gfpgan_resolution, interpolation=cv2.INTER_LINEAR)

            # Official normalization: BGR to RGB, normalize, transpose, normalize to [-1,1]
            img = img.astype(np.float32)[:,:,::-1] / 255.0  # BGR to RGB and [0,1]
            img = img.transpose((2, 0, 1))  # HWC to CHW
            img = (img - 0.5) / 0.5  # [0,1] to [-1,1]
            img = np.expand_dims(img, axis=0).astype(np.float32)  # Add batch dim

            # Run GFPGAN inference
            output = self.gfpgan_model.run(None, {'input': img})[0][0]

            # Official postprocess: transpose, clip, denormalize, RGB to BGR
            output = (output.transpose(1,2,0).clip(-1,1) + 1) * 0.5  # [-1,1] to [0,1]
            output = (output * 255)[:,:,::-1]  # [0,1] to [0,255] and RGB to BGR
            output = output.clip(0, 255).astype('uint8')

            # Resize back to original size if needed
            if output.shape[:2] != original_size:
                output = cv2.resize(output, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

            return output

        except Exception as e:
            print(f"Warning: GFPGAN enhancement failed: {e}")
            return face_img

    def detect_faces_and_align(self, frames):
        """
        Official face detection and alignment from wav2lip-onnx-HQ
        Returns aligned faces, crop faces, and transformation matrices
        """
        print("Detecting faces and generating alignment data...")

        crop_size = 256
        crop_faces = []
        sub_faces = []  # mouth regions
        matrices = []

        for i, frame in enumerate(tqdm(frames, desc="Face detection")):
            try:
                # Detect faces using RetinaFace (低い閾値で横顔・斜め顔も検出)
                bboxes, kpss = self.detector.detect(frame, input_size=(320, 320), det_thresh=0.1)

                if len(kpss) > 0:
                    # Use first detected face
                    kps = kpss[0]

                    # Get cropped and aligned face using official method
                    aligned_face, transformation_matrix = get_cropped_head_256(
                        frame, kps, size=crop_size, scale=1.0
                    )
                else:
                    # Fallback: create empty face
                    print(f"Warning: No face detected in frame {i}")
                    aligned_face = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                    transformation_matrix = np.float32([[1, 0, 0], [0, 1, 0]])

            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                aligned_face = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                transformation_matrix = np.float32([[1, 0, 0], [0, 1, 0]])

            # Extract mouth region using official coordinates
            padY = max(-15, min(self.pads, 15))
            if self.face_mode == 0:
                # Portrait mode
                sub_face = aligned_face[65-padY:241-padY, 62:194]
            else:
                # Square mode
                sub_face = aligned_face[65-padY:241-padY, 42:214]

            # Resize mouth region to model input size
            sub_face = cv2.resize(sub_face, (self.img_size, self.img_size))

            crop_faces.append(aligned_face)
            sub_faces.append(sub_face)
            matrices.append(transformation_matrix)

        return crop_faces, sub_faces, matrices

    def datagen(self, frames, mels, sub_faces):
        """
        Official data generator from wav2lip-onnx-HQ
        """
        img_batch, mel_batch = [], []
        frame_batch = []

        for i, mel_chunk in enumerate(mels):
            # Get corresponding frame (cycle if audio longer than video)
            frame_idx = i % len(frames)
            frame_to_save = frames[frame_idx].copy()
            frame_batch.append(frame_to_save)

            # Get face for current frame
            face = sub_faces[frame_idx]

            # Create masked version (official preprocessing)
            img_masked = face.copy()
            img_masked[self.img_size//2:] = 0  # Zero out bottom half (correct official way)

            # Concatenate masked and original (6 channels total)
            img_combined = np.concatenate((img_masked, face), axis=2)

            img_batch.append(img_combined)
            mel_batch.append(mel_chunk)

            # Process batch of 1 (can be increased for faster processing)
            if len(img_batch) >= 1:
                # Convert to arrays and normalize
                img_batch = np.asarray(img_batch) / 255.0
                mel_batch = np.asarray(mel_batch)

                # Transpose to model expected format (NCHW)
                img_batch = img_batch.transpose((0, 3, 1, 2)).astype(np.float32)
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                mel_batch = mel_batch.transpose((0, 3, 1, 2)).astype(np.float32)

                yield img_batch, mel_batch, frame_batch

                # Reset for next batch
                img_batch, mel_batch, frame_batch = [], [], []

    def process_video(self, video_path, audio_path, output_path):
        """Main processing function"""
        print(f"Processing video: {video_path}")
        print(f"Processing audio: {audio_path}")

        # Load video frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError("No frames found in video")

        print(f"Loaded {len(frames)} frames at {fps} FPS ({frame_width}x{frame_height})")

        # Process audio
        print("Processing audio...")
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel spectrogram contains NaN values')

        # Create mel chunks synchronized with video
        mel_chunks = []
        mel_idx_multiplier = 80.0 / fps

        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
            i += 1

        print(f"Generated {len(mel_chunks)} mel chunks")

        # 🔄 Loop video frames if audio is longer than video
        original_frame_count = len(frames)
        print(f"Original video frames: {original_frame_count}, Required: {len(mel_chunks)}")

        if len(mel_chunks) > len(frames):
            # Calculate how many loops we need
            loops_needed = (len(mel_chunks) - 1) // len(frames) + 1
            print(f"🔄 Looping video {loops_needed} times to match audio length")

            # Create looped frames list
            looped_frames = []
            for i in range(len(mel_chunks)):
                frame_idx = i % original_frame_count
                looped_frames.append(frames[frame_idx])
            frames = looped_frames
        else:
            # Limit frames to mel chunks length if video is longer
            frames = frames[:len(mel_chunks)]

        print(f"Final frame count: {len(frames)}, mel chunks: {len(mel_chunks)}")

        # Face detection and alignment
        crop_faces, sub_faces, matrices = self.detect_faces_and_align(frames)

        # Create blending mask for seamless integration
        static_face_mask = np.zeros((256, 256), dtype=np.uint8)
        static_face_mask = cv2.ellipse(static_face_mask, (128, 162), (62, 54), 0, 0, 360, (255, 255, 255), -1)
        static_face_mask = cv2.ellipse(static_face_mask, (128, 122), (46, 23), 0, 0, 360, (0, 0, 0), -1)
        static_face_mask = cv2.cvtColor(static_face_mask, cv2.COLOR_GRAY2RGB) / 255.0
        static_face_mask = cv2.GaussianBlur(static_face_mask, (19, 19), cv2.BORDER_DEFAULT)

        # Process frames through model (streaming for memory efficiency)
        print("Running Wav2Lip inference...")

        # Initialize streaming video writer (memory efficient)
        temp_video = "temp/temp_video.mp4"
        os.makedirs("temp", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_video, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        processed_frames = 0

        data_generator = self.datagen(frames, mel_chunks, sub_faces)
        total_batches = len(mel_chunks)

        for img_batch, mel_batch, frame_batch in tqdm(data_generator, total=total_batches, desc="Inference"):
            # Run model inference
            predictions = self.model.run(None, {
                'mel_spectrogram': mel_batch,
                'video_frames': img_batch
            })[0]

            # Process each prediction
            for i, pred in enumerate(predictions):
                # Convert CHW to HWC and denormalize
                pred_img = np.transpose(pred, (1, 2, 0)) * 255.0
                pred_img = np.clip(pred_img, 0, 255).astype(np.uint8)

                # Get corresponding frame data (with bounds checking)
                frame_idx = frame_count % original_frame_count  # Use original frame count for cycling
                original_frame = frames[frame_idx]
                crop_face = crop_faces[frame_idx].copy()
                transformation_matrix = matrices[frame_idx]

                # Resize prediction and place back in face
                padY = max(-15, min(self.pads, 15))
                if self.face_mode == 0:
                    mouth_h = (241 - padY) - (65 - padY)
                    mouth_w = 194 - 62
                    pred_resized = cv2.resize(pred_img, (mouth_w, mouth_h))
                    crop_face[65-padY:241-padY, 62:194] = pred_resized
                else:
                    mouth_h = (241 - padY) - (65 - padY)
                    mouth_w = 214 - 42
                    pred_resized = cv2.resize(pred_img, (mouth_w, mouth_h))
                    crop_face[65-padY:241-padY, 42:214] = pred_resized

                # Apply GFPGAN face enhancement if enabled
                if self.use_gfpgan:
                    crop_face = self.enhance_face_gfpgan(crop_face)

                # Transform back to original frame coordinate system
                frame_h, frame_w = original_frame.shape[:2]
                mat_inv = cv2.invertAffineTransform(transformation_matrix)

                # Warp face back to original position
                warped_face = cv2.warpAffine(crop_face, mat_inv, (frame_w, frame_h))
                warped_mask = cv2.warpAffine(static_face_mask, mat_inv, (frame_w, frame_h))

                # Blend with original frame
                result_frame = (warped_mask * warped_face.astype(np.float32) +
                               (1 - warped_mask) * original_frame.astype(np.float32))
                # Write frame directly to video (streaming - memory efficient)
                video_writer.write(result_frame.astype(np.uint8))
                processed_frames += 1

                frame_count += 1

                # Periodic GPU memory cleanup (every 50 frames)
                if processed_frames % 50 == 0:
                    import gc
                    gc.collect()

        # Close video writer (streaming completed)
        video_writer.release()
        print(f"Generated {processed_frames} result frames (streaming mode)")

        # Combine with audio using ffmpeg
        print("Adding audio to final video...")
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-shortest',
            output_path
        ]

        subprocess.run(cmd, check=True)

        # Clean up temp file
        if os.path.exists(temp_video):
            os.remove(temp_video)

        print(f"✓ Final video saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Wav2Lip with RetinaFace GPU')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to Wav2Lip model')
    parser.add_argument('--face', type=str, required=True, help='Path to input video')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--outfile', type=str, default='output/retinaface_result.mp4', help='Output path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--gfpgan', action='store_true', help='Enable GFPGAN face enhancement')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    os.makedirs('temp', exist_ok=True)

    print("=" * 70)
    print("🎬 Wav2Lip with RetinaFace GPU Detection")
    print("=" * 70)
    print(f"📹 Video: {args.face}")
    print(f"🎵 Audio: {args.audio}")
    print(f"🤖 Model: {args.checkpoint_path}")
    print(f"💾 Output: {args.outfile}")
    print(f"⚡ Device: {args.device}")
    print("=" * 70)

    try:
        processor = Wav2LipRetinaGPU(args.checkpoint_path, args.device, args.gfpgan)
        processor.process_video(args.face, args.audio, args.outfile)

        print("=" * 70)
        print("✅ SUCCESS! Video processing completed!")
        print(f"🎥 Result: {args.outfile}")
        print("=" * 70)

    except Exception as e:
        print("=" * 70)
        print("❌ ERROR during processing:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        sys.exit(1)

    finally:
        # Clean up GPU memory
        gc.collect()

if __name__ == "__main__":
    main()