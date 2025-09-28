#!/usr/bin/env python3
"""
Wav2Lip ONNX Correct Implementation
Fixed coordinate system and proper mouth region handling
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

class Wav2LipCorrect:
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
            return None

        # Get best detection
        best_detection = detections[0]
        x1, y1, x2, y2, confidence = best_detection

        return {
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': confidence
        }

    def get_face_region(self, frame, face_box):
        """Extract face region with proper padding"""
        x1, y1, x2, y2 = face_box

        # Calculate face dimensions
        face_width = x2 - x1
        face_height = y2 - y1

        # Add padding (20% on each side)
        pad_w = int(face_width * 0.2)
        pad_h = int(face_height * 0.2)

        # Expand region with padding
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(frame.shape[1], x2 + pad_w)
        y2 = min(frame.shape[0], y2 + pad_h)

        # Make it square
        w = x2 - x1
        h = y2 - y1
        size = max(w, h)

        # Center the square region
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        x1 = max(0, center_x - size // 2)
        y1 = max(0, center_y - size // 2)
        x2 = min(frame.shape[1], x1 + size)
        y2 = min(frame.shape[0], y1 + size)

        # Crop and resize to 256x256
        face_crop = frame[y1:y2, x1:x2]
        face_256 = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        return face_256, (x1, y1, x2, y2)

    def extract_lower_face_for_model(self, face_256):
        """
        Extract the lower face region that Wav2Lip expects
        This is the region that contains the mouth
        """
        # These are the EXACT coordinates from wav2lip-onnx-HQ
        # They extract the lower part of the face including mouth
        y1 = 65   # Start from nose area
        y2 = 248  # End at chin (183 pixels height)
        x1 = 62   # Left side
        x2 = 194  # Right side (132 pixels width)

        # Extract region (should be 183x132)
        lower_face = face_256[y1:y2, x1:x2]

        # Resize to 96x96 for model input
        lower_face_96 = cv2.resize(lower_face, (96, 96), interpolation=cv2.INTER_LANCZOS4)

        return lower_face_96, (x1, y1, x2, y2)

    def preprocess_for_model(self, face_96, reference_96):
        """Prepare input for Wav2Lip model"""
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

        # Denormalize from [-1, 1] to [0, 1]
        pred = (pred + 1.0) / 2.0
        pred = np.clip(pred, 0.0, 1.0)

        # Convert to uint8
        pred = (pred * 255.0).astype(np.uint8)

        return pred

    def place_mouth_in_face(self, face_256, mouth_prediction, mouth_coords):
        """Place the predicted mouth back in the face"""
        x1, y1, x2, y2 = mouth_coords

        # Resize prediction from 96x96 to original mouth region size
        mouth_h = y2 - y1  # Should be 183
        mouth_w = x2 - x1  # Should be 132
        mouth_resized = cv2.resize(mouth_prediction, (mouth_w, mouth_h), interpolation=cv2.INTER_LANCZOS4)

        # Create smooth blending mask
        mask = np.ones((mouth_h, mouth_w), dtype=np.float32)

        # Feather the edges for smooth blending
        feather_size = 8
        for i in range(feather_size):
            alpha = (i + 1) / feather_size
            # Top edge
            if i < mouth_h:
                mask[i, :] = alpha
            # Bottom edge
            if mouth_h - i - 1 >= 0:
                mask[mouth_h - i - 1, :] = alpha
            # Left edge
            if i < mouth_w:
                mask[:, i] = np.minimum(mask[:, i], alpha)
            # Right edge
            if mouth_w - i - 1 >= 0:
                mask[:, mouth_w - i - 1] = np.minimum(mask[:, mouth_w - i - 1], alpha)

        # Apply mask for blending
        mask_3d = np.stack([mask, mask, mask], axis=2)
        original_mouth = face_256[y1:y2, x1:x2].astype(np.float32)
        new_mouth = mouth_resized.astype(np.float32)

        # Blend
        blended = original_mouth * (1 - mask_3d) + new_mouth * mask_3d

        # Place back
        result_face = face_256.copy()
        result_face[y1:y2, x1:x2] = blended.astype(np.uint8)

        return result_face

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

        # Extend frames if audio is longer
        if len(mel_chunks) > len(frames):
            num_loops = (len(mel_chunks) + len(frames) - 1) // len(frames)
            extended_frames = frames * num_loops
            extended_frames = extended_frames[:len(mel_chunks)]
        else:
            extended_frames = frames[:len(mel_chunks)]

        print(f"Generated {len(mel_chunks)} mel chunks")

        # Get reference face from first frame
        first_face = None
        for frame in frames:
            first_face = self.detect_face(frame)
            if first_face:
                break

        if not first_face:
            raise RuntimeError("No face detected in video")

        print(f"Reference face detected with confidence: {first_face['confidence']:.3f}")

        # Get reference face image
        ref_face_256, _ = self.get_face_region(frames[0], first_face['box'])
        ref_lower_face_96, _ = self.extract_lower_face_for_model(ref_face_256)

        # Process frames
        result_frames = []

        for i, (frame, mel_chunk) in enumerate(tqdm(zip(extended_frames, mel_chunks),
                                                    desc="Processing", total=len(mel_chunks))):
            # Detect face in current frame
            face_info = self.detect_face(frame)
            if not face_info:
                # Use first face as fallback
                face_info = first_face

            # Get face region
            face_256, face_coords = self.get_face_region(frame, face_info['box'])

            # Extract lower face for model
            lower_face_96, mouth_coords = self.extract_lower_face_for_model(face_256)

            # Prepare input
            input_batch = self.preprocess_for_model(lower_face_96, ref_lower_face_96)
            mel_batch = mel_chunk[np.newaxis, np.newaxis, :, :].astype(np.float32)

            # Run inference
            predictions = self.wav2lip_model.run(None, {
                'video_frames': input_batch,
                'mel_spectrogram': mel_batch
            })

            # Get prediction
            prediction = predictions[0][0]

            # Postprocess
            mouth_image = self.postprocess_prediction(prediction)

            # Place mouth back in face
            result_face = self.place_mouth_in_face(face_256, mouth_image, mouth_coords)

            # Place face back in frame
            x1, y1, x2, y2 = face_coords
            face_h = y2 - y1
            face_w = x2 - x1

            # Resize face to original size
            face_resized = cv2.resize(result_face, (face_w, face_h), interpolation=cv2.INTER_LANCZOS4)

            # Place in frame
            result_frame = frame.copy()
            result_frame[y1:y2, x1:x2] = face_resized

            result_frames.append(result_frame)

        # Save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in result_frames:
            out.write(frame)
        out.release()

        print(f"✓ Video saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Wav2Lip ONNX Correct Implementation')
    parser.add_argument('video', help='Input video file')
    parser.add_argument('audio', help='Input audio file')
    parser.add_argument('-o', '--output', default='output/correct_result.mp4', help='Output video file')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--models', default='models', help='Models directory')

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(exist_ok=True)

    print("=" * 60)
    print("🎬 Wav2Lip ONNX Correct Implementation")
    print("=" * 60)
    print(f"📹 Video: {args.video}")
    print(f"🎵 Audio: {args.audio}")
    print(f"💾 Output: {args.output}")
    print(f"⚡ Device: {args.device}")
    print("=" * 60)

    try:
        # Initialize processor
        processor = Wav2LipCorrect(args.models, args.device)

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