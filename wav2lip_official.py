#!/usr/bin/env python3
"""
Wav2Lip ONNX Official Style Implementation
Based on wav2lip-onnx-HQ approach
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

class Wav2LipOfficial:
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
        """Detect face using RetinaFace - Official style"""
        detections, landmarks = self.face_detector.detect(frame, threshold=0.3)

        if len(detections) == 0:
            raise RuntimeError("No face detected in frame")

        # Get best detection
        best_detection = detections[0]
        x1, y1, x2, y2, confidence = best_detection

        return {
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': confidence
        }

    def get_cropped_head_256(self, frame, face_box):
        """
        Official wav2lip-onnx-HQ style face cropping
        Returns 256x256 face image
        """
        x1, y1, x2, y2 = face_box

        # Calculate face dimensions
        face_width = x2 - x1
        face_height = y2 - y1

        # Expand face region (wav2lip-onnx-HQ style)
        expansion_factor = 1.3  # 30% expansion
        expanded_width = int(face_width * expansion_factor)
        expanded_height = int(face_height * expansion_factor)

        # Calculate center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Make it square using the larger dimension
        crop_size = max(expanded_width, expanded_height)

        # Calculate crop coordinates
        half_size = crop_size // 2
        crop_x1 = max(0, center_x - half_size)
        crop_y1 = max(0, center_y - half_size)
        crop_x2 = min(frame.shape[1], center_x + half_size)
        crop_y2 = min(frame.shape[0], center_y + half_size)

        # Crop face
        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize to 256x256
        face_256 = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        return face_256, (crop_x1, crop_y1, crop_x2, crop_y2)

    def get_face_image_96(self, face_256):
        """
        Extract face region for Wav2Lip input
        Official wav2lip-onnx-HQ coordinates
        """
        # Official coordinates from wav2lip-onnx-HQ
        # Bottom half of face + mouth region
        padY = 0
        y1 = 65 - padY   # Start from upper lip area
        y2 = 248 - padY  # Go down to include lower face
        x1 = 62          # Left side
        x2 = 194         # Right side

        # Extract the region
        face_region = face_256[y1:y2, x1:x2]  # Should be 183x132

        # Resize to 96x96 for Wav2Lip
        face_96 = cv2.resize(face_region, (96, 96), interpolation=cv2.INTER_LANCZOS4)

        return face_96

    def preprocess_frames(self, frames_96):
        """Preprocess frames for Wav2Lip model"""
        batch = []
        for frame in frames_96:
            # Normalize to [-1, 1]
            normalized = (frame.astype(np.float32) / 255.0 - 0.5) / 0.5
            batch.append(normalized)

        # Stack and convert to NCHW format
        batch = np.stack(batch, axis=0)
        batch = np.transpose(batch, (0, 3, 1, 2))

        return batch.astype(np.float32)

    def process_video(self, video_path, audio_path, output_path):
        """Main processing function - Official wav2lip-onnx-HQ style"""
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

        # Create mel chunks (official wav2lip-onnx-HQ approach)
        mel_chunks = []
        mel_step_size = 16
        mel_idx_multiplier = 80.0 / fps

        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > mel.shape[1]:
                mel_chunks.append(mel[:, mel.shape[1] - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
            i += 1

        print(f"Generated {len(mel_chunks)} mel chunks")

        # Detect face in first frame
        first_face = self.detect_face(frames[0])
        print(f"Face detected with confidence: {first_face['confidence']:.3f}")

        # Get first frame face crop for reference
        first_face_256, first_crop_coords = self.get_cropped_head_256(frames[0], first_face['box'])
        reference_face_96 = self.get_face_image_96(first_face_256)

        # Process frames in batches (official approach)
        batch_size = 128
        result_frames = []

        # Extend frames to match mel chunks (loop video)
        num_frames_needed = len(mel_chunks)
        extended_frames = []
        for i in range(num_frames_needed):
            extended_frames.append(frames[i % len(frames)])

        for batch_start in tqdm(range(0, len(mel_chunks), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(mel_chunks))

            # Prepare batch data
            img_batch = []
            mel_batch = []
            frame_data = []

            for i in range(batch_start, batch_end):
                frame = extended_frames[i]
                mel_chunk = mel_chunks[i]

                # Detect face or use reference
                try:
                    face_info = self.detect_face(frame)
                    face_256, crop_coords = self.get_cropped_head_256(frame, face_info['box'])
                except:
                    # Use first frame as fallback
                    face_256, crop_coords = self.get_cropped_head_256(frame, first_face['box'])

                # Get current face for inference
                current_face_96 = self.get_face_image_96(face_256)

                # Prepare input: concatenate current + reference (6 channels)
                combined_face = np.concatenate([current_face_96, reference_face_96], axis=2)
                img_batch.append(combined_face)
                mel_batch.append(mel_chunk)
                frame_data.append((frame, face_256, crop_coords))

            # Convert to model format
            img_batch = self.preprocess_frames(img_batch)
            mel_batch = np.array(mel_batch, dtype=np.float32)[:, np.newaxis, :, :]  # Add channel dimension

            # Run inference
            predictions = self.wav2lip_model.run(None, {
                'video_frames': img_batch,
                'mel_spectrogram': mel_batch
            })[0]

            # Process predictions
            for j, (pred, (frame, face_256, crop_coords)) in enumerate(zip(predictions, frame_data)):
                # Convert prediction back to image
                pred_img = np.transpose(pred, (1, 2, 0))  # CHW to HWC
                pred_img = np.clip((pred_img + 1.0) / 2.0, 0.0, 1.0)  # [-1,1] to [0,1]
                pred_img = (pred_img * 255.0).astype(np.uint8)

                # Resize prediction to match face region
                pred_resized = cv2.resize(pred_img, (132, 183), interpolation=cv2.INTER_LANCZOS4)

                # Place prediction back in face_256
                padY = 0
                y1, y2 = 65 - padY, 248 - padY
                x1, x2 = 62, 194

                # Create blending mask
                mask = np.ones((y2-y1, x2-x1), dtype=np.float32)
                # Feather edges
                for k in range(5):
                    alpha = (k + 1) / 5.0
                    mask[k, :] = alpha
                    mask[-(k+1), :] = alpha
                    mask[:, k] = np.minimum(mask[:, k], alpha)
                    mask[:, -(k+1)] = np.minimum(mask[:, -(k+1)], alpha)

                # Apply mask
                mask_3d = np.stack([mask, mask, mask], axis=2)
                original_region = face_256[y1:y2, x1:x2].astype(np.float32)
                pred_float = pred_resized.astype(np.float32)

                blended = original_region * (1 - mask_3d) + pred_float * mask_3d
                face_256[y1:y2, x1:x2] = blended.astype(np.uint8)

                # Place face back in original frame
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
                crop_h = crop_y2 - crop_y1
                crop_w = crop_x2 - crop_x1

                face_resized = cv2.resize(face_256, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)

                result_frame = frame.copy()
                result_frame[crop_y1:crop_y2, crop_x1:crop_x2] = face_resized
                result_frames.append(result_frame)

        # Save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in result_frames:
            out.write(frame)
        out.release()

        print(f"✓ Video saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Wav2Lip ONNX Official Style')
    parser.add_argument('video', help='Input video file')
    parser.add_argument('audio', help='Input audio file')
    parser.add_argument('-o', '--output', default='output/official_result.mp4', help='Output video file')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--models', default='models', help='Models directory')

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(exist_ok=True)

    print("=" * 60)
    print("🎬 Wav2Lip ONNX Official Style Implementation")
    print("=" * 60)
    print(f"📹 Video: {args.video}")
    print(f"🎵 Audio: {args.audio}")
    print(f"💾 Output: {args.output}")
    print(f"⚡ Device: {args.device}")
    print("=" * 60)

    try:
        # Initialize processor
        processor = Wav2LipOfficial(args.models, args.device)

        # Process video
        processor.process_video(args.video, args.audio, args.output)

        print("=" * 60)
        print("✅ SUCCESS! Lip-sync generation completed!")
        print(f"🎥 Output saved: {args.output}")
        print("=" * 60)

    except Exception as e:
        print("=" * 60)
        print("❌ ERROR occurred during processing:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()