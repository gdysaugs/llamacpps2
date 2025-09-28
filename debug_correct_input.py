#!/usr/bin/env python3
"""
Debug Correct Input Preprocessing
Test with proper reference frame and mel preprocessing
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'utils'))

import onnxruntime as ort
from utils.retinaface_pytorch import RetinaFaceDetector
import src.audio as audio

def debug_correct_preprocessing():
    # Load models
    wav2lip_path = 'models/wav2lip/wav2lip_gan.onnx'
    retinaface_path = 'models/detection_Resnet50_Final.pth'

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    wav2lip_model = ort.InferenceSession(wav2lip_path, providers=providers)
    face_detector = RetinaFaceDetector(retinaface_path, device='cuda')

    # Load test video and audio
    video_path = 'input/target_video_3s.mp4'
    audio_path = 'input/「ばかーー！」.mp3'

    # Load multiple frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(10):  # Load first 10 frames
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Loaded {len(frames)} frames")

    # Process audio with correct preprocessing
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)

    # Get first mel chunk
    first_mel_chunk = mel[:, 0:16]

    # Wav2Lip expects mel in specific format
    print(f"Original mel shape: {mel.shape}")
    print(f"Mel chunk shape: {first_mel_chunk.shape}")
    print(f"Mel range: {first_mel_chunk.min():.3f} to {first_mel_chunk.max():.3f}")

    def extract_face_96x96(frame, face_detector):
        """Extract face similar to wav2lip-onnx-HQ style"""
        detections, landmarks = face_detector.detect(frame, threshold=0.3)
        if len(detections) == 0:
            return None

        x1, y1, x2, y2, confidence = detections[0]

        # Calculate center and size
        face_w = x2 - x1
        face_h = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Create square crop with padding
        size = max(face_w, face_h) * 1.3
        half_size = size / 2

        crop_x1 = max(0, int(center_x - half_size))
        crop_y1 = max(0, int(center_y - half_size))
        crop_x2 = min(frame.shape[1], int(center_x + half_size))
        crop_y2 = min(frame.shape[0], int(center_y + half_size))

        # Ensure square
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        crop_size = min(crop_w, crop_h)
        crop_x2 = crop_x1 + crop_size
        crop_y2 = crop_y1 + crop_size

        # Crop and resize to 256x256
        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        face_256 = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # Extract lower face region (mouth area) like wav2lip-onnx-HQ
        # Coordinates from official implementation
        padY = 0
        sub_face = face_256[65-padY:241-padY, 62:194]  # 176x132 region

        # Resize to 96x96
        face_96 = cv2.resize(sub_face, (96, 96), interpolation=cv2.INTER_LANCZOS4)

        return face_96, confidence

    # Extract reference face (first frame)
    ref_face, ref_conf = extract_face_96x96(frames[0], face_detector)
    if ref_face is None:
        print("No face in reference frame")
        return

    print(f"Reference face confidence: {ref_conf:.3f}")
    cv2.imwrite("debug_ref_face.jpg", ref_face)

    # Extract current face (different frame)
    current_face, curr_conf = extract_face_96x96(frames[5], face_detector)  # Use frame 5
    if current_face is None:
        current_face = ref_face  # Fallback
        curr_conf = ref_conf

    print(f"Current face confidence: {curr_conf:.3f}")
    cv2.imwrite("debug_current_face.jpg", current_face)

    # Correct preprocessing
    def preprocess_face(face_img):
        """Correct face preprocessing for Wav2Lip"""
        # Convert to float32 and normalize to [-1, 1]
        face_float = face_img.astype(np.float32) / 255.0
        face_norm = (face_float - 0.5) / 0.5
        return face_norm

    def preprocess_mel(mel_chunk):
        """Correct mel preprocessing"""
        return mel_chunk.astype(np.float32)

    # Preprocess inputs
    ref_norm = preprocess_face(ref_face)
    curr_norm = preprocess_face(current_face)
    mel_processed = preprocess_mel(first_mel_chunk)

    print(f"Ref face range: {ref_norm.min():.3f} to {ref_norm.max():.3f}")
    print(f"Current face range: {curr_norm.min():.3f} to {curr_norm.max():.3f}")
    print(f"Mel range: {mel_processed.min():.3f} to {mel_processed.max():.3f}")

    # Create proper 6-channel input (reference + current)
    # NOTE: Order might be important - try both ways
    input_variants = [
        ("ref_first", np.concatenate([ref_norm, curr_norm], axis=2)),
        ("curr_first", np.concatenate([curr_norm, ref_norm], axis=2))
    ]

    for variant_name, combined_face in input_variants:
        print(f"\nTesting variant: {variant_name}")

        # Add batch dimension and transpose to NCHW
        video_input = combined_face[np.newaxis]
        video_input = np.transpose(video_input, (0, 3, 1, 2))

        # Mel input
        mel_input = mel_processed[np.newaxis, np.newaxis, :, :]

        print(f"Video input shape: {video_input.shape}")
        print(f"Mel input shape: {mel_input.shape}")

        try:
            # Run inference
            predictions = wav2lip_model.run(None, {
                'video_frames': video_input.astype(np.float32),
                'mel_spectrogram': mel_input.astype(np.float32)
            })

            prediction = predictions[0][0]
            print(f"Output range: {prediction.min():.3f} to {prediction.max():.3f}")

            # Convert to image
            pred_hwc = np.transpose(prediction, (1, 2, 0))

            # Try different post-processing
            if prediction.max() <= 1.0 and prediction.min() >= 0.0:
                # Already in [0, 1] range
                pred_img = (pred_hwc * 255).astype(np.uint8)
            else:
                # Assume [-1, 1] range
                pred_img = ((pred_hwc + 1.0) / 2.0 * 255).astype(np.uint8)

            cv2.imwrite(f"debug_result_{variant_name}.jpg", pred_img)
            print(f"Saved: debug_result_{variant_name}.jpg")

        except Exception as e:
            print(f"Error with {variant_name}: {e}")

    print("\n✓ Testing complete. Check debug_result_*.jpg files.")

if __name__ == "__main__":
    debug_correct_preprocessing()