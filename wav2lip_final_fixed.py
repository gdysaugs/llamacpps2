#!/usr/bin/env python3
"""
Wav2Lip ONNX Final Fixed Implementation
Based on exact wav2lip-onnx-HQ coordinate system and affine transformation
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

class Wav2LipFinalFixed:
    def __init__(self, model_path, device='cuda', face_mode=0):
        self.device = device
        self.model_path = model_path
        self.img_size = 96
        self.face_mode = face_mode  # 0=portrait (original), 1=square
        self.padY = 0  # Mouth position adjustment

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

    def detect_face_with_matrix(self, frame):
        """Detect face and return transformation matrix like wav2lip-onnx-HQ"""
        detections, landmarks = self.face_detector.detect(frame, threshold=0.1)

        if len(detections) == 0:
            return None, None, None

        # Get best detection
        best_detection = detections[0]
        x1, y1, x2, y2, confidence = best_detection

        # Calculate face center and dimensions
        face_w = x2 - x1
        face_h = y2 - y1
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # Create crop region (square, 256x256)
        size = max(face_w, face_h) * 1.3  # 30% padding like original
        crop_size = 256

        # Calculate crop coordinates
        crop_x1 = max(0, int(center_x - size / 2))
        crop_y1 = max(0, int(center_y - size / 2))
        crop_x2 = min(frame.shape[1], int(center_x + size / 2))
        crop_y2 = min(frame.shape[0], int(center_y + size / 2))

        # Ensure square
        actual_w = crop_x2 - crop_x1
        actual_h = crop_y2 - crop_y1
        actual_size = min(actual_w, actual_h)
        crop_x2 = crop_x1 + actual_size
        crop_y2 = crop_y1 + actual_size

        # Crop and resize to 256x256
        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        face_256 = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # Create transformation matrix for later warping back
        # From 256x256 crop back to original position
        scale = actual_size / 256.0
        M = np.array([
            [scale, 0, crop_x1],
            [0, scale, crop_y1]
        ], dtype=np.float32)

        # Inverse matrix for warping back
        M_inv = cv2.invertAffineTransform(M)

        return face_256, M, M_inv

    def extract_sub_face(self, face_256):
        """Extract mouth region exactly like wav2lip-onnx-HQ"""
        if self.face_mode == 0:
            # Original portrait mode
            sub_face = face_256[65-self.padY:241-self.padY, 62:194]
        else:
            # Square mode for less mouth opening
            sub_face = face_256[65-self.padY:241-self.padY, 42:214]

        # Resize to model input size
        sub_face = cv2.resize(sub_face, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)

        return sub_face

    def create_sub_face_mask(self):
        """Create soft mask for mouth region blending"""
        mask = np.zeros((256, 256, 3), dtype=np.float32)

        if self.face_mode == 0:
            # Portrait mode coordinates
            cv2.rectangle(mask, (62, 65-self.padY), (194, 241-self.padY), (255, 255, 255), -1)
        else:
            # Square mode coordinates
            cv2.rectangle(mask, (42, 65-self.padY), (214, 241-self.padY), (255, 255, 255), -1)

        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (29, 29), cv2.BORDER_DEFAULT)
        mask = mask / 255.0

        return mask

    def preprocess_for_model(self, face_96, reference_96):
        """Preprocess for Wav2Lip model"""
        # Normalize to [-1, 1]
        face_norm = (face_96.astype(np.float32) / 255.0 - 0.5) / 0.5
        ref_norm = (reference_96.astype(np.float32) / 255.0 - 0.5) / 0.5

        # Concatenate (6 channels)
        combined = np.concatenate([face_norm, ref_norm], axis=2)

        # Add batch dimension and transpose to NCHW
        batch = combined[np.newaxis]
        batch = np.transpose(batch, (0, 3, 1, 2))

        return batch.astype(np.float32)

    def postprocess_prediction(self, prediction):
        """Convert model output back to image"""
        # CHW to HWC
        pred = np.transpose(prediction, (1, 2, 0))

        # Denormalize from [-1, 1] to [0, 255]
        pred = (pred + 1.0) / 2.0
        pred = np.clip(pred, 0.0, 1.0)
        pred = (pred * 255.0).astype(np.uint8)

        return pred

    def place_prediction_in_face(self, face_256, prediction_96):
        """Place 96x96 prediction back into 256x256 face using exact coordinates"""
        result_face = face_256.copy()

        # Resize prediction to match mouth region size
        if self.face_mode == 0:
            # Portrait mode: 62:194, 65-padY:241-padY
            mouth_w = 194 - 62  # 132 pixels
            mouth_h = (241 - self.padY) - (65 - self.padY)  # 176 pixels
            mouth_region = cv2.resize(prediction_96, (mouth_w, mouth_h), interpolation=cv2.INTER_LANCZOS4)
            result_face[65-self.padY:241-self.padY, 62:194] = mouth_region
        else:
            # Square mode: 42:214, 65-padY:241-padY
            mouth_w = 214 - 42  # 172 pixels
            mouth_h = (241 - self.padY) - (65 - self.padY)  # 176 pixels
            mouth_region = cv2.resize(prediction_96, (mouth_w, mouth_h), interpolation=cv2.INTER_LANCZOS4)
            result_face[65-self.padY:241-self.padY, 42:214] = mouth_region

        return result_face

    def warp_back_to_frame(self, processed_face, original_frame, M):
        """Warp processed face back to original frame using affine transformation"""
        frame_h, frame_w = original_frame.shape[:2]

        # Warp the processed face back to original frame coordinates
        warped_face = cv2.warpAffine(processed_face, M, (frame_w, frame_h))

        # Create warped mask
        face_mask = np.ones((256, 256, 3), dtype=np.uint8) * 255
        warped_mask = cv2.warpAffine(face_mask, M, (frame_w, frame_h))
        warped_mask = warped_mask.astype(np.float32) / 255.0

        # Apply Gaussian blur to mask for soft blending
        warped_mask = cv2.GaussianBlur(warped_mask, (15, 15), cv2.BORDER_DEFAULT)

        # Blend
        result = (warped_mask * warped_face.astype(np.float32) +
                 (1 - warped_mask) * original_frame.astype(np.float32))

        return result.astype(np.uint8)

    def process_video(self, video_path, audio_path, output_path):
        """Main processing function"""
        print(f"Processing: {video_path} + {audio_path}")
        print(f"Face mode: {self.face_mode} ({'portrait' if self.face_mode == 0 else 'square'})")

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

        # Extend frames if audio is longer
        if len(mel_chunks) > len(frames):
            num_loops = (len(mel_chunks) + len(frames) - 1) // len(frames)
            extended_frames = frames * num_loops
            extended_frames = extended_frames[:len(mel_chunks)]
        else:
            extended_frames = frames[:len(mel_chunks)]

        print(f"Generated {len(mel_chunks)} mel chunks")

        # Get reference face from first frame
        ref_face_256, _, _ = self.detect_face_with_matrix(frames[0])
        if ref_face_256 is None:
            raise RuntimeError("No face detected in first frame")

        ref_sub_face = self.extract_sub_face(ref_face_256)
        print("Reference face extracted successfully")

        # Process frames
        result_frames = []

        for i, (frame, mel_chunk) in enumerate(tqdm(zip(extended_frames, mel_chunks),
                                                    desc="Processing", total=len(mel_chunks))):
            # Detect face and get transformation matrix
            face_256, M, M_inv = self.detect_face_with_matrix(frame)
            if face_256 is None:
                # If no face detected, use original frame
                result_frames.append(frame)
                continue

            # Extract mouth region
            current_sub_face = self.extract_sub_face(face_256)

            # Prepare input for model
            input_batch = self.preprocess_for_model(current_sub_face, ref_sub_face)
            mel_batch = mel_chunk[np.newaxis, np.newaxis, :, :].astype(np.float32)

            # Run inference
            predictions = self.wav2lip_model.run(None, {
                'video_frames': input_batch,
                'mel_spectrogram': mel_batch
            })

            # Process prediction
            prediction = predictions[0][0]
            prediction_img = self.postprocess_prediction(prediction)

            # Place prediction back in face
            result_face = self.place_prediction_in_face(face_256, prediction_img)

            # Warp back to original frame using affine transformation
            final_frame = self.warp_back_to_frame(result_face, frame, M)

            result_frames.append(final_frame)

        # Save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in result_frames:
            out.write(frame)
        out.release()

        print(f"✓ Video saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Wav2Lip ONNX Final Fixed Implementation')
    parser.add_argument('video', help='Input video file')
    parser.add_argument('audio', help='Input audio file')
    parser.add_argument('-o', '--output', default='output/final_fixed_result.mp4', help='Output video file')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--models', default='models', help='Models directory')
    parser.add_argument('--face-mode', type=int, default=0, choices=[0, 1],
                       help='Face mode: 0=portrait (original), 1=square (less mouth opening)')

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(exist_ok=True)

    print("=" * 60)
    print("🎬 Wav2Lip ONNX Final Fixed Implementation")
    print("=" * 60)
    print(f"📹 Video: {args.video}")
    print(f"🎵 Audio: {args.audio}")
    print(f"💾 Output: {args.output}")
    print(f"⚡ Device: {args.device}")
    print(f"👤 Face mode: {args.face_mode}")
    print("=" * 60)

    try:
        # Initialize processor
        processor = Wav2LipFinalFixed(args.models, args.device, args.face_mode)

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