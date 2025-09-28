#!/usr/bin/env python3
import sys
import os
import argparse
import cv2
import numpy as np
import onnxruntime as ort
import librosa
from pathlib import Path
from tqdm import tqdm
import subprocess

class Wav2LipOnnxHQ:
    def __init__(self, models_dir, device='cuda'):
        self.models_dir = Path(models_dir)
        self.device = device
        self.img_size = 96
        self.mel_step_size = 16
        
        # Set providers
        if device == 'cuda':
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load wav2lip ONNX model"""
        wav2lip_path = self.models_dir / 'wav2lip' / 'wav2lip_gan.onnx'
        
        if not wav2lip_path.exists():
            raise FileNotFoundError(f"Wav2Lip model not found: {wav2lip_path}")
        
        print(f"Loading Wav2Lip model: {wav2lip_path.name}...")
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.wav2lip_model = ort.InferenceSession(str(wav2lip_path), session_options, providers=self.providers)
        print(f"✓ Wav2Lip model loaded with providers: {self.wav2lip_model.get_providers()}")
        
    def get_mel_spectrogram(self, audio_path):
        """Generate mel spectrogram from audio"""
        print("Processing audio...")
        
        # Load audio
        wav, sr = librosa.load(audio_path, sr=16000)
        
        # Generate mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_fft=800, hop_length=200, n_mels=80
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        
        if mel.shape[1] > 1:
            mel = mel[:, :-1]  # Remove last frame
            
        # Normalize
        mel = (mel + 5) / 5
        
        return mel.T  # (frames, mel_features)
        
    def chunk_mel_spectrogram(self, mel, fps):
        """Chunk mel spectrogram to match video frames"""
        mel_chunks = []
        mel_idx_multiplier = 80.0 / fps
        i = 0
        
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel):
                mel_chunks.append(mel[len(mel) - self.mel_step_size:])
                break
            mel_chunks.append(mel[start_idx:start_idx + self.mel_step_size])
            i += 1
            
        return mel_chunks
        
    def load_video_frames(self, video_path):
        """Load video frames"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        return frames, fps, (width, height)
        
    def detect_face(self, frame):
        """Simple face detection using OpenCV"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        if len(faces) > 0:
            # Get largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Expand face region
            padding = int(max(w, h) * 0.3)
            size = max(w, h) + padding
            
            x_center = x + w // 2
            y_center = y + h // 2
            
            x_new = max(0, x_center - size // 2)
            y_new = max(0, y_center - size // 2)
            
            return [x_new, y_new, size, size]
        else:
            # Default center region
            h, w = frame.shape[:2]
            return [w//4, h//4, w//2, h//2]
            
    def crop_face_for_wav2lip(self, frame, face_rect):
        """Crop and process face for wav2lip exactly like wav2lip-onnx-HQ"""
        x, y, w, h = [int(v) for v in face_rect]
        
        # Ensure bounds
        frame_h, frame_w = frame.shape[:2]
        x_end = min(frame_w, x + w)
        y_end = min(frame_h, y + h)
        x = max(0, x)
        y = max(0, y)
        
        # Crop face region
        face_region = frame[y:y_end, x:x_end]
        
        if face_region.size == 0:
            return None, None, None
        
        # Resize to 256x256 (aligned face)
        aligned_face = cv2.resize(face_region, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        # Extract mouth region [65:241, 62:194] = 176x132
        padY = 0
        mouth_region = aligned_face[65-padY:241-padY, 62:194]
        
        # Resize to model input size (96x96)
        mouth_resized = cv2.resize(mouth_region, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        
        return mouth_resized, (x, y, x_end - x, y_end - y), aligned_face
        
    def datagen(self, frames, mel_chunks):
        """Data generator exactly like wav2lip-onnx-HQ"""
        img_batch, mel_batch, frame_batch = [], [], []
        
        for i, m in enumerate(mel_chunks):
            idx = i % len(frames)  # non-static mode
            
            frame_to_save = frames[idx].copy()
            frame_batch.append(frame_to_save)
            
            img_batch.append(frames[idx])
            mel_batch.append(m)
            
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            
            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0  # Mask right half
            
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            
            yield img_batch, mel_batch, frame_batch
            img_batch, mel_batch, frame_batch = [], [], []
            
    def inference(self, video_path, audio_path, output_path):
        """Main inference function following wav2lip-onnx-HQ exactly"""
        
        # Load video
        frames, fps, (orig_width, orig_height) = self.load_video_frames(video_path)
        
        # Generate mel spectrogram
        mel = self.get_mel_spectrogram(audio_path)
        mel_chunks = self.chunk_mel_spectrogram(mel, fps)
        
        print(f"Generated {len(mel_chunks)} mel chunks")
        
        # Trim frames to match mel chunks
        frames = frames[:len(mel_chunks)]
        
        # Detect faces and crop
        sub_faces = []
        coords_list = []
        aligned_faces = []
        
        for frame in frames:
            face_rect = self.detect_face(frame)
            sub_face, coords, aligned_face = self.crop_face_for_wav2lip(frame, face_rect)
            
            if sub_face is not None:
                sub_faces.append(sub_face)
                coords_list.append(coords)
                aligned_faces.append(aligned_face)
            else:
                # Fallback
                h, w = frame.shape[:2]
                center_region = frame[h//4:3*h//4, w//4:3*w//4]
                aligned_face = cv2.resize(center_region, (256, 256), interpolation=cv2.INTER_CUBIC)
                
                padY = 0
                mouth_region = aligned_face[65-padY:241-padY, 62:194]
                sub_face = cv2.resize(mouth_region, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
                
                sub_faces.append(sub_face)
                coords_list.append((w//4, h//4, w//2, h//2))
                aligned_faces.append(aligned_face)
        
        # Generate lip-sync
        print("Generating lip-sync...")
        gen = self.datagen(sub_faces, mel_chunks)
        
        output_frames = []
        
        for i, (img_batch, mel_batch, frames_batch) in enumerate(tqdm(gen, total=len(mel_chunks))):
            
            # Wav2lip-onnx-HQ exact inference
            img_batch = img_batch.transpose((0, 3, 1, 2)).astype(np.float32)
            mel_batch = mel_batch.transpose((0, 3, 2, 1)).astype(np.float32)  # Fix mel dimensions
            
            pred = self.wav2lip_model.run(None, {
                'mel_spectrogram': mel_batch, 
                'video_frames': img_batch
            })[0][0]
            
            # Wav2lip-onnx-HQ exact post-processing
            pred = pred.transpose(1, 2, 0) * 255
            pred = pred.astype(np.uint8)
            
            # Process result
            orig_frame = frames[min(i, len(frames)-1)]
            safe_idx = min(i, len(coords_list)-1)
            x, y, w, h = coords_list[safe_idx]
            aligned_face = aligned_faces[safe_idx].copy()
            
            # Resize prediction and place into aligned face
            padY = 0
            pred_resized = cv2.resize(pred, (132, 176))  # wav2lip-onnx-HQ exact
            aligned_face[65-padY:241-padY, 62:194] = pred_resized
            
            # Resize back to original face size and place in frame
            face_final = cv2.resize(aligned_face, (w, h), interpolation=cv2.INTER_CUBIC)
            result_frame = orig_frame.copy()
            result_frame[y:y+h, x:x+w] = face_final
            
            output_frames.append(result_frame)
        
        # Write output video
        self.write_video_with_audio(output_frames, audio_path, output_path, fps)
        print(f"✓ Output saved: {output_path}")
        
    def write_video_with_audio(self, frames, audio_path, output_path, fps):
        """Write video with audio"""
        import uuid
        temp_video = str(Path(output_path).parent / f'temp_{uuid.uuid4().hex[:8]}.mp4')
        
        # Write video frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Merge with audio
        cmd = [
            'ffmpeg', '-i', temp_video, '-i', audio_path,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '5',  # High quality
            '-preset', 'slow',
            '-c:a', 'libmp3lame', '-ac', '2', '-ar', '44100', '-ab', '128000',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', output_path, '-y'
        ]
        
        subprocess.run(cmd, capture_output=True)
        
        # Cleanup
        if os.path.exists(temp_video):
            os.remove(temp_video)

def main():
    parser = argparse.ArgumentParser(description='Wav2Lip ONNX-HQ Official Inference')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('audio_path', help='Path to input audio file')
    parser.add_argument('-o', '--output', default='output/result_onnx_hq_official.mp4', 
                        help='Output video path')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎬 Wav2Lip ONNX-HQ Official Inference")
    print("=" * 60)
    print(f"📹 Video: {args.video_path}")
    print(f"🎵 Audio: {args.audio_path}")
    print(f"💾 Output: {args.output}")
    print(f"⚡ Device: {args.device}")
    print("=" * 60)
    
    try:
        # Get script directory for models
        script_dir = Path(__file__).parent
        models_dir = script_dir / 'models'
        
        wav2lip = Wav2LipOnnxHQ(str(models_dir), device=args.device)
        
        # Create output directory
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        
        # Run inference
        wav2lip.inference(
            video_path=args.video_path,
            audio_path=args.audio_path,
            output_path=args.output
        )
        
        print("=" * 60)
        print("✅ SUCCESS! Lip-sync generation completed!")
        print(f"🎥 Output saved: {args.output}")
        print("=" * 60)
        
    except Exception as e:
        import traceback
        print("=" * 60)
        print("❌ ERROR occurred during processing:")
        print(f"Error: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()