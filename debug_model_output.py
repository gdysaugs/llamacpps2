#!/usr/bin/env python3
"""
Debug Wav2Lip Model Output
Check if the model is producing valid predictions
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

def debug_model():
    # Load models
    wav2lip_path = 'models/wav2lip/wav2lip_gan.onnx'
    retinaface_path = 'models/detection_Resnet50_Final.pth'

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    wav2lip_model = ort.InferenceSession(wav2lip_path, providers=providers)
    face_detector = RetinaFaceDetector(retinaface_path, device='cuda')

    # Load test video and audio
    video_path = 'input/target_video_3s.mp4'
    audio_path = 'input/「ばかーー！」.mp3'

    # Load first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to load video")
        return

    # Process audio
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    first_mel_chunk = mel[:, 0:16]

    print(f"Frame shape: {frame.shape}")
    print(f"Mel chunk shape: {first_mel_chunk.shape}")

    # Detect face
    detections, landmarks = face_detector.detect(frame, threshold=0.3)
    if len(detections) == 0:
        print("No face detected")
        return

    x1, y1, x2, y2, confidence = detections[0]
    print(f"Face detected: confidence={confidence:.3f}, box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

    # Simple face crop to 96x96
    face_w = x2 - x1
    face_h = y2 - y1
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    size = max(face_w, face_h) * 1.3
    half_size = int(size / 2)

    crop_x1 = max(0, center_x - half_size)
    crop_y1 = max(0, center_y - half_size)
    crop_x2 = min(frame.shape[1], center_x + half_size)
    crop_y2 = min(frame.shape[0], center_y + half_size)

    face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    face_resized = cv2.resize(face_crop, (96, 96), interpolation=cv2.INTER_LANCZOS4)

    # Save input face
    cv2.imwrite("debug_input_face.jpg", face_resized)
    print("Input face saved: debug_input_face.jpg")

    # Prepare model input
    # Normalize to [-1, 1]
    face_norm = (face_resized.astype(np.float32) / 255.0 - 0.5) / 0.5

    # Duplicate for 6 channels (current + reference)
    face_6ch = np.concatenate([face_norm, face_norm], axis=2)

    # Add batch dimension and transpose to NCHW
    video_input = face_6ch[np.newaxis]
    video_input = np.transpose(video_input, (0, 3, 1, 2))

    # Prepare mel input
    mel_input = first_mel_chunk[np.newaxis, np.newaxis, :, :]

    print(f"Video input shape: {video_input.shape}")
    print(f"Mel input shape: {mel_input.shape}")
    print(f"Video input range: {video_input.min():.3f} to {video_input.max():.3f}")
    print(f"Mel input range: {mel_input.min():.3f} to {mel_input.max():.3f}")

    # Run model
    try:
        predictions = wav2lip_model.run(None, {
            'video_frames': video_input.astype(np.float32),
            'mel_spectrogram': mel_input.astype(np.float32)
        })

        prediction = predictions[0][0]  # First batch, first output
        print(f"Model output shape: {prediction.shape}")
        print(f"Model output range: {prediction.min():.3f} to {prediction.max():.3f}")

        # Convert CHW to HWC
        pred_hwc = np.transpose(prediction, (1, 2, 0))

        # Check different normalization methods
        methods = [
            ("raw", pred_hwc),
            ("clip_only", np.clip(pred_hwc, 0, 1)),
            ("normalize_pm1", (pred_hwc + 1.0) / 2.0),
            ("normalize_pm1_clip", np.clip((pred_hwc + 1.0) / 2.0, 0, 1)),
            ("tanh_like", (np.tanh(pred_hwc) + 1.0) / 2.0)
        ]

        for method_name, processed in methods:
            # Convert to uint8
            img_uint8 = (np.clip(processed, 0, 1) * 255).astype(np.uint8)

            # Save
            cv2.imwrite(f"debug_output_{method_name}.jpg", img_uint8)
            print(f"Saved: debug_output_{method_name}.jpg (range: {processed.min():.3f} to {processed.max():.3f})")

        print("\n✓ Debug complete. Check the debug_output_*.jpg files to see which normalization works.")

    except Exception as e:
        print(f"Model inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model()