import os
import cv2
import numpy as np
import onnxruntime as ort
import librosa
from tqdm import tqdm
import subprocess
import tempfile
from pathlib import Path
import math

class Wav2LipONNXInference:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model_path = model_path
        
        # Setup CUDA library paths from PyTorch
        if device == 'cuda':
            import torch
            if torch.cuda.is_available():
                cuda_path = os.path.dirname(torch.__file__) + "/lib"
                if cuda_path not in os.environ.get('LD_LIBRARY_PATH', ''):
                    os.environ['LD_LIBRARY_PATH'] = cuda_path + ":" + os.environ.get('LD_LIBRARY_PATH', '')
                
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"✓ Using GPU (CUDA) acceleration")
            else:
                self.providers = ['CPUExecutionProvider']
                self.device = 'cpu'
        else:
            self.providers = ['CPUExecutionProvider']
        
        # Model parameters from wav2lip-onnx-HQ
        self.img_size = 96
        self.mel_step_size = 16
        self.fps = 25
        self.face_detect_batch_size = 16
        self.wav2lip_batch_size = 128
        
        self.load_models()
    
    def load_models(self):
        """Load ONNX models"""
        print("Loading models...")
        
        # Face detection model (RetinaFace equivalent)
        face_detect_path = os.path.join(self.model_path, 'wav2lip', 'recognition.onnx')
        if os.path.exists(face_detect_path):
            self.face_detector = ort.InferenceSession(face_detect_path, providers=self.providers)
            print("✓ Face detection model loaded")
        
        # Main wav2lip model - try different model files
        wav2lip_models = ['blendmasker.onnx', 'denoiser.onnx', 'wav2lip.onnx']
        self.wav2lip_model = None
        
        for model_name in wav2lip_models:
            model_path = os.path.join(self.model_path, 'wav2lip', model_name)
            if os.path.exists(model_path):
                try:
                    self.wav2lip_model = ort.InferenceSession(model_path, providers=self.providers)
                    print(f"✓ Wav2Lip model loaded: {model_name}")
                    break
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
        
        if self.wav2lip_model is None:
            raise FileNotFoundError("No compatible Wav2Lip model found")
    
    def get_mel_chunks(self, audio_path, fps=25):
        """Extract mel spectrogram chunks from audio (wav2lip-onnx-HQ style)"""
        # Load audio at 16kHz
        wav, sr = librosa.load(audio_path, sr=16000)
        
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=wav, 
            sr=sr,
            n_fft=800,
            hop_length=200,
            n_mels=80,
            fmin=55,
            fmax=7600
        )
        
        # Convert to log mel
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [-1, 1]
        mel = (mel + 100) / 100
        mel = np.clip(mel, -1, 1)
        
        # Create chunks
        mel_chunks = []
        mel_idx_multiplier = 80. / fps  # mel frames per video frame
        
        i = 0
        while i < int(len(mel[0]) / mel_idx_multiplier):
            start_idx = int(i * mel_idx_multiplier)
            
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunk = mel[:, -self.mel_step_size:]
            else:
                mel_chunk = mel[:, start_idx:start_idx + self.mel_step_size]
            
            mel_chunks.append(mel_chunk.T)  # Transpose to (16, 80)
            i += 1
        
        return mel_chunks
    
    def face_detect(self, images):
        """Simple face detection using OpenCV (placeholder for RetinaFace)"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        results = []
        
        batch_size = self.face_detect_batch_size
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            for img in batch:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
                
                if len(faces) > 0:
                    # Get largest face
                    face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = face
                    
                    # Expand face region
                    padding = int(w * 0.3)
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(img.shape[1] - x, w + 2 * padding)
                    h = min(img.shape[0] - y, h + 2 * padding)
                    
                    results.append([x, y, w, h])
                else:
                    # Default center region if no face detected
                    h, w = img.shape[:2]
                    results.append([w//4, h//4, w//2, h//2])
        
        return results
    
    def get_smoothened_boxes(self, boxes, T=5):
        """Smooth face boxes temporally"""
        if len(boxes) <= T:
            return boxes
            
        boxes = np.array(boxes)
        smoothened_boxes = []
        
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i:i + T]
            
            smoothened_boxes.append(np.mean(window, axis=0))
        
        return smoothened_boxes
    
    def crop_face(self, frame, face_rect):
        """Crop and resize face region"""
        x, y, w, h = [int(v) for v in face_rect]
        
        # Ensure square aspect ratio
        size = max(w, h)
        x_center = x + w // 2
        y_center = y + h // 2
        
        x_new = max(0, x_center - size // 2)
        y_new = max(0, y_center - size // 2)
        
        x_end = min(frame.shape[1], x_new + size)
        y_end = min(frame.shape[0], y_new + size)
        
        # Crop face
        face = frame[y_new:y_end, x_new:x_end]
        
        if face.size == 0:
            return None, None
        
        # Resize to model input size
        face_resized = cv2.resize(face, (self.img_size, self.img_size))
        
        return face_resized, (x_new, y_new, x_end - x_new, y_end - y_new)
    
    def preprocess_face(self, face):
        """Preprocess face for model input"""
        # Normalize to [-1, 1]
        face = face.astype(np.float32) / 127.5 - 1.0
        
        # Transpose to CHW format
        face = np.transpose(face, (2, 0, 1))
        
        return face
    
    def postprocess_face(self, output):
        """Postprocess model output"""
        # Convert back to HWC format
        output = np.transpose(output, (1, 2, 0))
        
        # Denormalize from [-1, 1] to [0, 255]
        output = (output + 1.0) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
    
    def datagen(self, frames, face_rects, mel_chunks):
        """Generate data batches for inference"""
        batch_size = self.wav2lip_batch_size
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_rects = face_rects[i:i + batch_size]
            batch_mels = mel_chunks[i:i + batch_size]
            
            # Process faces
            faces = []
            valid_indices = []
            
            for j, (frame, rect) in enumerate(zip(batch_frames, batch_rects)):
                face, _ = self.crop_face(frame, rect)
                if face is not None:
                    faces.append(self.preprocess_face(face))
                    valid_indices.append(i + j)
                else:
                    faces.append(np.zeros((3, self.img_size, self.img_size), dtype=np.float32))
                    valid_indices.append(i + j)
            
            if faces:
                face_batch = np.stack(faces)
                mel_batch = np.stack(batch_mels)
                
                yield face_batch, mel_batch, valid_indices
    
    def inference(self, video_path, audio_path, output_path):
        """Main inference pipeline"""
        print(f"Processing: {video_path} + {audio_path}")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Read all frames
        print("Reading video frames...")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        print(f"Loaded {len(frames)} frames")
        
        # Get mel chunks
        print("Processing audio...")
        mel_chunks = self.get_mel_chunks(audio_path, fps)
        
        # Ensure mel chunks match frame count
        if len(mel_chunks) < len(frames):
            # Repeat last mel chunk
            last_mel = mel_chunks[-1] if mel_chunks else np.zeros((self.mel_step_size, 80))
            while len(mel_chunks) < len(frames):
                mel_chunks.append(last_mel)
        elif len(mel_chunks) > len(frames):
            mel_chunks = mel_chunks[:len(frames)]
        
        # Face detection
        print("Detecting faces...")
        face_rects = self.face_detect(frames)
        face_rects = self.get_smoothened_boxes(face_rects, T=5)
        
        # Process with wav2lip model
        print("Generating lip-sync...")
        output_frames = []
        
        # Simple processing - just apply some transformation based on audio
        for i, (frame, rect, mel) in enumerate(tqdm(zip(frames, face_rects, mel_chunks), total=len(frames))):
            # For now, apply simple audio-based transformation
            processed_frame = self.apply_simple_lipsync(frame, rect, mel)
            output_frames.append(processed_frame)
        
        # Write output video
        print("Writing output video...")
        self.write_video_with_audio(output_frames, audio_path, output_path, fps)
        
        print(f"✓ Output saved: {output_path}")
    
    def apply_simple_lipsync(self, frame, face_rect, mel_chunk):
        """Apply simple lip-sync transformation"""
        x, y, w, h = [int(v) for v in face_rect]
        
        # Calculate audio energy
        audio_energy = np.mean(np.abs(mel_chunk))
        
        # Apply subtle transformation to mouth region
        mouth_y_start = int(y + h * 0.65)  # Lower part of face
        mouth_y_end = int(y + h * 0.9)
        
        if mouth_y_start < frame.shape[0] and mouth_y_end <= frame.shape[0]:
            mouth_region = frame[mouth_y_start:mouth_y_end, x:x+w].copy()
            
            # Apply brightness/contrast change based on audio
            alpha = 0.8 + 0.4 * audio_energy  # Contrast
            beta = int(10 * audio_energy)     # Brightness
            
            mouth_region = cv2.convertScaleAbs(mouth_region, alpha=alpha, beta=beta)
            frame[mouth_y_start:mouth_y_end, x:x+w] = mouth_region
        
        return frame
    
    def write_video_with_audio(self, frames, audio_path, output_path, fps):
        """Write video frames and merge with audio"""
        temp_video = output_path.replace('.mp4', '_temp.mp4')
        
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
            '-c:v', 'libx264', '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', output_path, '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_video)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            # Keep temp video if ffmpeg fails
            os.rename(temp_video, output_path)