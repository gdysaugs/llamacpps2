#!/usr/bin/env python3
"""
Simple Wav2Lip ONNX Inference
Based on wav2lip-onnx-HQ official implementation
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import onnxruntime as ort

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'utils'))

# Import modules
from utils.retinaface_pytorch import RetinaFaceDetector
import src.audio as audio

class SimpleWav2Lip:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model_path = model_path
        self.img_size = 96

        # Set up ONNX providers
        if device == 'cuda':
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']

        # Load models
        self.load_wav2lip_model()
        self.load_face_detector()

    def load_wav2lip_model(self):
        """Load Wav2Lip ONNX model"""
        wav2lip_path = os.path.join(self.model_path, 'wav2lip', 'wav2lip_gan.onnx')
        if not os.path.exists(wav2lip_path):
            raise FileNotFoundError(f"Wav2Lip model not found: {wav2lip_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.wav2lip_model = ort.InferenceSession(
            wav2lip_path,
            sess_options=session_options,
            providers=self.providers
        )
        print(f"✓ Wav2Lip model loaded: {wav2lip_path}")

    def load_face_detector(self):
        """Load RetinaFace detector"""
        retinaface_path = os.path.join(self.model_path, 'detection_Resnet50_Final.pth')
        if not os.path.exists(retinaface_path):
            raise FileNotFoundError(f"RetinaFace model not found: {retinaface_path}")

        self.face_detector = RetinaFaceDetector(retinaface_path, device=self.device)
        print(f"✓ RetinaFace model loaded: {retinaface_path}")

    def detect_face(self, frame):
        """Detect face using RetinaFace"""
        detections, landmarks = self.face_detector.detect(frame, threshold=0.3)

        if len(detections) == 0:
            raise RuntimeError("No face detected in frame. Please ensure the face is clearly visible.")

        # Get best detection (highest confidence)
        best_detection = detections[0]
        best_landmarks = landmarks[0]

        x1, y1, x2, y2, confidence = best_detection

        # Validate coordinates
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"Invalid face detection coordinates: ({x1}, {y1}, {x2}, {y2})")

        return {
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'landmarks': best_landmarks.reshape(-1, 2),
            'confidence': confidence
        }

    def crop_face(self, frame, face_info):
        """Crop face from frame - Fixed coordinate calculation"""
        x1, y1, x2, y2 = face_info['box']

        # Make face region square by expanding to largest dimension
        face_w = x2 - x1
        face_h = y2 - y1
        face_size = max(face_w, face_h)

        # Calculate center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Create square crop region (no extra padding to keep coordinates simple)
        half_size = face_size // 2
        crop_x1 = max(0, center_x - half_size)
        crop_y1 = max(0, center_y - half_size)
        crop_x2 = min(frame.shape[1], center_x + half_size)
        crop_y2 = min(frame.shape[0], center_y + half_size)

        # Ensure we have a square region
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        crop_size = min(crop_w, crop_h)

        # Adjust to make it perfectly square
        crop_x2 = crop_x1 + crop_size
        crop_y2 = crop_y1 + crop_size

        # Crop face region
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize to 256x256 for processing
        face_256 = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # Use the ENTIRE face_256 as input (not just mouth region)
        # Wav2Lip expects the full lower face, not just mouth
        face_96 = cv2.resize(face_256, (96, 96), interpolation=cv2.INTER_LANCZOS4)

        return {
            'face_256': face_256,
            'face_96': face_96,  # Use full face, not just mouth
            'crop_coords': (crop_x1, crop_y1, crop_x2, crop_y2)
        }

    def preprocess_input(self, mouth_image, reference_mouth, mel_chunk):
        """Preprocess input for Wav2Lip model"""
        # Normalize both images to [-1, 1]
        mouth_normalized = (mouth_image.astype(np.float32) / 255.0 - 0.5) / 0.5
        ref_normalized = (reference_mouth.astype(np.float32) / 255.0 - 0.5) / 0.5

        # Concatenate current and reference frames (6 channels total)
        combined = np.concatenate([mouth_normalized, ref_normalized], axis=2)

        # Prepare batch (1, H, W, C) -> (1, C, H, W)
        mouth_batch = np.transpose(combined[np.newaxis], (0, 3, 1, 2))

        # Prepare mel spectrogram
        mel_batch = mel_chunk[np.newaxis, np.newaxis]  # (1, 1, 80, 16)

        return mouth_batch.astype(np.float32), mel_batch.astype(np.float32)

    def postprocess_output(self, prediction):
        """Postprocess Wav2Lip output"""
        # Convert from CHW to HWC and denormalize
        pred_hwc = np.transpose(prediction, (1, 2, 0))
        pred_denorm = np.clip((pred_hwc + 1.0) / 2.0, 0.0, 1.0)
        pred_uint8 = (pred_denorm * 255.0).astype(np.uint8)

        return pred_uint8

    def blend_mouth(self, face_256, new_mouth, mouth_coords):
        """Blend new mouth into face"""
        mouth_x1, mouth_y1, mouth_x2, mouth_y2 = mouth_coords

        # Create blending mask with soft edges
        mask = np.ones((mouth_y2 - mouth_y1, mouth_x2 - mouth_x1), dtype=np.float32)

        # Feather edges (3 pixel blend)
        for i in range(3):
            alpha = (i + 1) / 3.0
            mask[i, :] = alpha
            mask[-(i+1), :] = alpha
            mask[:, i] = np.minimum(mask[:, i], alpha)
            mask[:, -(i+1)] = np.minimum(mask[:, -(i+1)], alpha)

        # Apply mask
        mask_3d = np.stack([mask, mask, mask], axis=2)
        original_mouth = face_256[mouth_y1:mouth_y2, mouth_x1:mouth_x2].astype(np.float32)
        new_mouth_float = new_mouth.astype(np.float32)

        blended = original_mouth * (1 - mask_3d) + new_mouth_float * mask_3d
        face_256[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = blended.astype(np.uint8)

        return face_256

    def place_face_in_frame(self, original_frame, processed_face, crop_coords):
        """Place processed face back into original frame"""
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords

        # Resize processed face to match crop size
        crop_h = crop_y2 - crop_y1
        crop_w = crop_x2 - crop_x1
        face_resized = cv2.resize(processed_face, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)

        # Place back in frame
        result_frame = original_frame.copy()
        result_frame[crop_y1:crop_y2, crop_x1:crop_x2] = face_resized

        return result_frame

    def process_video(self, video_path, audio_path, output_path):
        """Main processing function"""
        print(f"Processing: {video_path} + {audio_path}")

        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        print(f"Loaded {len(frames)} frames at {fps} FPS ({width}x{height})")

        # Process audio
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)

        # Create mel chunks
        mel_chunks = []
        mel_step_size = 16
        mel_idx_multiplier = 80.0 / fps

        for i in range(len(frames)):
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > mel.shape[1]:
                mel_chunks.append(mel[:, mel.shape[1] - mel_step_size:])
            else:
                mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])

        print(f"Generated {len(mel_chunks)} mel chunks")

        # Detect face in first frame and get reference mouth
        first_face = self.detect_face(frames[0])
        reference_crop = self.crop_face(frames[0], first_face)
        reference_mouth = reference_crop['mouth_96']
        print(f"Face detected with confidence: {first_face['confidence']:.3f}")

        # Process frames
        result_frames = []

        for i, (frame, mel_chunk) in enumerate(tqdm(zip(frames, mel_chunks), desc="Processing frames", total=len(frames))):
            # Detect face in current frame (or use first frame as reference)
            try:
                face_info = self.detect_face(frame)
            except:
                # Use first frame face as fallback
                face_info = first_face

            # Crop face and mouth
            crop_data = self.crop_face(frame, face_info)

            # Preprocess for Wav2Lip (current mouth + reference mouth)
            mouth_batch, mel_batch = self.preprocess_input(
                crop_data['mouth_96'],
                reference_mouth,
                mel_chunk
            )

            # Run Wav2Lip inference
            prediction = self.wav2lip_model.run(None, {
                'video_frames': mouth_batch,
                'mel_spectrogram': mel_batch
            })[0][0]

            # Postprocess prediction
            new_mouth = self.postprocess_output(prediction)

            # Blend mouth into face
            result_face = self.blend_mouth(
                crop_data['face_256'].copy(),
                new_mouth,
                crop_data['mouth_coords']
            )

            # Place face back in frame
            result_frame = self.place_face_in_frame(frame, result_face, crop_data['crop_coords'])
            result_frames.append(result_frame)

        # Save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in result_frames:
            out.write(frame)
        out.release()

        print(f"✓ Video saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Simple Wav2Lip ONNX Inference')
    parser.add_argument('video', help='Input video file')
    parser.add_argument('audio', help='Input audio file')
    parser.add_argument('-o', '--output', default='output/simple_result.mp4', help='Output video file')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--models', default='models', help='Models directory')

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(exist_ok=True)

    print("=" * 50)
    print("🎬 Simple Wav2Lip ONNX Inference")
    print("=" * 50)
    print(f"📹 Video: {args.video}")
    print(f"🎵 Audio: {args.audio}")
    print(f"💾 Output: {args.output}")
    print(f"⚡ Device: {args.device}")
    print("=" * 50)

    try:
        # Initialize processor
        processor = SimpleWav2Lip(args.models, args.device)

        # Process video
        processor.process_video(args.video, args.audio, args.output)

        print("=" * 50)
        print("✅ SUCCESS! Lip-sync generation completed!")
        print(f"🎥 Output saved: {args.output}")
        print("=" * 50)

    except Exception as e:
        print("=" * 50)
        print("❌ ERROR occurred during processing:")
        print(f"Error: {e}")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main()