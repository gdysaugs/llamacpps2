#!/usr/bin/env python3
"""
Direct adaptation of wav2lip-onnx-HQ official implementation
Using GAN model and RetinaFace detection
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

class Wav2LipOfficialDirect:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model_path = model_path
        self.img_size = 96
        self.face_mode = 0  # 0=portrait (62:194), 1=square (42:214)
        self.padY = 0

        # ONNX providers
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
        print(f"✓ Wav2Lip GAN model loaded: {wav2lip_path}")

    def load_face_detector(self):
        """Load RetinaFace detector"""
        retinaface_path = os.path.join(self.model_path, 'detection_Resnet50_Final.pth')
        if not os.path.exists(retinaface_path):
            raise FileNotFoundError(f"RetinaFace model not found: {retinaface_path}")

        self.face_detector = RetinaFaceDetector(retinaface_path, device=self.device)
        print(f"✓ RetinaFace model loaded: {retinaface_path}")

    def face_detect(self, images):
        """
        Official wav2lip-onnx-HQ face detection logic
        Returns crop_faces, sub_faces, matrix for transformation
        """
        crop_faces = []
        sub_faces = []
        matrix = []

        for image in images:
            # Detect faces
            detections, landmarks = self.face_detector.detect(image, threshold=0.3)

            if len(detections) == 0:
                # Fallback: use center region
                h, w = image.shape[:2]
                # Create a default 256x256 crop from center
                center_x, center_y = w // 2, h // 2
                size = min(w, h) // 2
                x1, y1 = max(0, center_x - size), max(0, center_y - size)
                x2, y2 = min(w, center_x + size), min(h, center_y + size)
                crop_face = image[y1:y2, x1:x2]
                crop_face = cv2.resize(crop_face, (256, 256), interpolation=cv2.INTER_LANCZOS4)

                # Create identity matrix (no transformation)
                M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            else:
                # Use detected face
                x1, y1, x2, y2, confidence = detections[0]

                # Calculate face center and dimensions
                face_w = x2 - x1
                face_h = y2 - y1
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                # Create square crop region with padding
                size = max(face_w, face_h) * 1.3  # 30% padding
                half_size = size / 2

                crop_x1 = max(0, int(center_x - half_size))
                crop_y1 = max(0, int(center_y - half_size))
                crop_x2 = min(image.shape[1], int(center_x + half_size))
                crop_y2 = min(image.shape[0], int(center_y + half_size))

                # Ensure square
                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1
                crop_size = min(crop_w, crop_h)
                crop_x2 = crop_x1 + crop_size
                crop_y2 = crop_y1 + crop_size

                # Crop and resize to 256x256
                face_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_face = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

                # Create transformation matrix for warping back
                scale = crop_size / 256.0
                M = np.array([
                    [scale, 0.0, crop_x1],
                    [0.0, scale, crop_y1]
                ], dtype=np.float32)

            # Extract mouth region using official coordinates
            if self.face_mode == 0:
                # Portrait mode coordinates
                sub_face = crop_face[65-self.padY:241-self.padY, 62:194]
            else:
                # Square mode coordinates
                sub_face = crop_face[65-self.padY:241-self.padY, 42:214]

            # Resize to model input size
            sub_face = cv2.resize(sub_face, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)

            crop_faces.append(crop_face)
            sub_faces.append(sub_face)
            matrix.append(M)

        return crop_faces, sub_faces, matrix

    def datagen(self, mels, frames):
        """
        Official wav2lip-onnx-HQ data generator
        """
        img_batch, mel_batch = [], []
        frame_h, frame_w = frames[0].shape[:-1]

        # Get face data
        crop_faces, sub_faces, matrix = self.face_detect(frames)

        # Use first frame as reference
        ref_face = sub_faces[0]

        for i, m in enumerate(mels):
            frame_idx = i % len(frames)
            current_face = sub_faces[frame_idx]

            # Prepare current face
            current_face = current_face / 255.0
            current_face = (current_face - 0.5) / 0.5  # Normalize to [-1, 1]

            # Prepare reference face
            ref_face_norm = ref_face / 255.0
            ref_face_norm = (ref_face_norm - 0.5) / 0.5  # Normalize to [-1, 1]

            # Combine current and reference (6 channels)
            combined_face = np.concatenate([current_face, ref_face_norm], axis=2)

            img_batch.append(combined_face)
            mel_batch.append(m)

            if len(img_batch) >= 128:  # Batch size
                # Convert to numpy arrays
                img_batch = np.array(img_batch)
                mel_batch = np.array(mel_batch)

                # Transpose to NCHW format
                img_batch = img_batch.transpose((0, 3, 1, 2))
                mel_batch = mel_batch[:, np.newaxis, :, :]

                yield img_batch.astype(np.float32), mel_batch.astype(np.float32), crop_faces, matrix

                img_batch, mel_batch = [], []

        # Process remaining batch
        if len(img_batch) > 0:
            img_batch = np.array(img_batch)
            mel_batch = np.array(mel_batch)

            img_batch = img_batch.transpose((0, 3, 1, 2))
            mel_batch = mel_batch[:, np.newaxis, :, :]

            yield img_batch.astype(np.float32), mel_batch.astype(np.float32), crop_faces, matrix

    def process_video(self, video_path, audio_path, output_path):
        """Main processing function using official logic"""
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

        # Process audio using original audio.py
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)

        # Create mel chunks
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

        # Process using official datagen
        result_frames = []
        batch_count = 0

        for img_batch, mel_batch, crop_faces, matrix in tqdm(self.datagen(mel_chunks, frames), desc="Processing"):
            batch_count += 1

            # Run inference
            predictions = self.wav2lip_model.run(None, {
                'video_frames': img_batch,
                'mel_spectrogram': mel_batch
            })

            pred_batch = predictions[0]

            # Process each prediction in batch
            for i, prediction in enumerate(pred_batch):
                # Convert CHW to HWC
                pred_img = np.transpose(prediction, (1, 2, 0))

                # The model outputs values in [0, 1] range
                pred_img = np.clip(pred_img, 0.0, 1.0)
                pred_img = (pred_img * 255.0).astype(np.uint8)

                # Get corresponding crop face and matrix
                frame_idx = (len(result_frames)) % len(frames)
                crop_face = crop_faces[frame_idx].copy()
                M = matrix[frame_idx]

                # Resize prediction to mouth region size
                if self.face_mode == 0:
                    mouth_h = (241 - self.padY) - (65 - self.padY)  # 176
                    mouth_w = 194 - 62  # 132
                    pred_resized = cv2.resize(pred_img, (mouth_w, mouth_h), interpolation=cv2.INTER_LANCZOS4)
                    crop_face[65-self.padY:241-self.padY, 62:194] = pred_resized
                else:
                    mouth_h = (241 - self.padY) - (65 - self.padY)  # 176
                    mouth_w = 214 - 42  # 172
                    pred_resized = cv2.resize(pred_img, (mouth_w, mouth_h), interpolation=cv2.INTER_LANCZOS4)
                    crop_face[65-self.padY:241-self.padY, 42:214] = pred_resized

                # Warp back to original frame using transformation matrix
                original_frame = frames[frame_idx]
                frame_h, frame_w = original_frame.shape[:2]

                # Warp the processed face back
                warped_face = cv2.warpAffine(crop_face, M, (frame_w, frame_h))

                # Create mask for blending
                face_mask = np.ones((256, 256, 3), dtype=np.uint8) * 255
                warped_mask = cv2.warpAffine(face_mask, M, (frame_w, frame_h))
                warped_mask = cv2.GaussianBlur(warped_mask, (15, 15), cv2.BORDER_DEFAULT)
                warped_mask = warped_mask.astype(np.float32) / 255.0

                # Blend
                result_frame = (warped_mask * warped_face.astype(np.float32) +
                               (1 - warped_mask) * original_frame.astype(np.float32))

                result_frames.append(result_frame.astype(np.uint8))

        print(f"Processed {len(result_frames)} frames in {batch_count} batches")

        # Save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in result_frames:
            out.write(frame)
        out.release()

        print(f"✓ Video saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Wav2Lip ONNX Official Direct Implementation')
    parser.add_argument('video', help='Input video file')
    parser.add_argument('audio', help='Input audio file')
    parser.add_argument('-o', '--output', default='output/official_direct_result.mp4', help='Output video file')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--models', default='models', help='Models directory')

    args = parser.parse_args()

    Path(args.output).parent.mkdir(exist_ok=True)

    print("=" * 60)
    print("🎬 Wav2Lip ONNX Official Direct Implementation")
    print("=" * 60)
    print(f"📹 Video: {args.video}")
    print(f"🎵 Audio: {args.audio}")
    print(f"💾 Output: {args.output}")
    print(f"⚡ Device: {args.device}")
    print("=" * 60)

    try:
        processor = Wav2LipOfficialDirect(args.models, args.device)
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