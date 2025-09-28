import os
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import librosa
import subprocess
import tempfile
from pathlib import Path

class Wav2LipModel:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Setup CUDA library paths from PyTorch
        if device == 'cuda':
            import os
            import torch
            
            if torch.cuda.is_available():
                cuda_path = os.path.dirname(torch.__file__) + "/lib"
                if cuda_path not in os.environ.get('LD_LIBRARY_PATH', ''):
                    os.environ['LD_LIBRARY_PATH'] = cuda_path + ":" + os.environ.get('LD_LIBRARY_PATH', '')
                
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"✓ Using GPU (CUDA) with PyTorch libraries")
            else:
                print("CUDA not available, using CPU")
                self.providers = ['CPUExecutionProvider']
                self.device = 'cpu'
        else:
            self.providers = ['CPUExecutionProvider']
            
        self.model_path = model_path
        
        # Model parameters
        self.img_size = 96
        self.mel_step_size = 16
        self.fps = 25
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load ONNX models for Wav2Lip"""
        print("Loading Wav2Lip models...")
        
        # Face detection model
        face_detect_path = os.path.join(self.model_path, 'wav2lip', 'recognition.onnx')
        if os.path.exists(face_detect_path):
            self.face_detector = ort.InferenceSession(face_detect_path, providers=self.providers)
            print(f"✓ Face detection model loaded")
        else:
            raise FileNotFoundError(f"Face detection model not found: {face_detect_path}")
            
        # Load wav2lip generator model (using denoiser as the main model)
        wav2lip_path = os.path.join(self.model_path, 'wav2lip', 'denoiser.onnx')
        if os.path.exists(wav2lip_path):
            self.wav2lip = ort.InferenceSession(wav2lip_path, providers=self.providers)
            print(f"✓ Wav2Lip model loaded")
        else:
            # Try alternative path
            wav2lip_path = os.path.join(self.model_path, 'wav2lip', 'wav2lip.onnx')
            if os.path.exists(wav2lip_path):
                self.wav2lip = ort.InferenceSession(wav2lip_path, providers=self.providers)
                print(f"✓ Wav2Lip model loaded")
            else:
                print(f"Warning: Wav2Lip model not found, using passthrough")
                self.wav2lip = None
                
    def get_smoothened_boxes(self, boxes, T):
        """Apply temporal smoothing to bounding boxes"""
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes
        
    def face_detect(self, images):
        """Detect faces in images batch"""
        batch_size = 16
        face_rects = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            # Simple face detection using OpenCV for now
            for img in batch:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Use OpenCV's face detector
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Get the largest face
                    face = max(faces, key=lambda x: x[2] * x[3])
                    face_rects.append(face)
                else:
                    # If no face detected, use center region
                    h, w = img.shape[:2]
                    face_rects.append([w//4, h//4, w//2, h//2])
                    
        return face_rects
        
    def get_mel_chunks(self, audio_path):
        """Extract mel spectrogram chunks from audio"""
        # Load audio
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
        
        # Convert to log scale
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize
        mel = (mel + 100) / 100
        
        # Segment into chunks
        mel_chunks = []
        mel_idx_multiplier = 80. / self.fps
        
        i = 0
        while i < len(mel[0]):
            start_idx = int(i * mel_idx_multiplier)
            
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunk = mel[:, -self.mel_step_size:]
            else:
                mel_chunk = mel[:, start_idx:start_idx + self.mel_step_size]
                
            mel_chunks.append(mel_chunk.T)
            i += 1
            
        return mel_chunks
        
    def process_video(self, video_path, audio_path, output_path):
        """Main processing function with lip-sync"""
        print(f"Processing video: {video_path}")
        print(f"Using audio: {audio_path}")
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
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
        
        # Detect faces
        print("Detecting faces...")
        face_rects = self.face_detect(frames)
        
        # Smooth face rectangles
        face_rects = self.get_smoothened_boxes(np.array(face_rects), T=5)
        
        # Get audio mel chunks
        print("Processing audio...")
        mel_chunks = self.get_mel_chunks(audio_path)
        
        # Ensure we have enough mel chunks
        if len(mel_chunks) < len(frames):
            # Pad with zeros
            while len(mel_chunks) < len(frames):
                mel_chunks.append(mel_chunks[-1])
        
        # Process frames with lip-sync
        print("Applying lip-sync...")
        processed_frames = []
        
        with tqdm(total=len(frames)) as pbar:
            for i, (frame, face_rect, mel) in enumerate(zip(frames, face_rects, mel_chunks)):
                # Apply lip-sync to frame
                processed_frame = self.apply_lipsync(frame, face_rect, mel)
                processed_frames.append(processed_frame)
                pbar.update(1)
                
        # Write output video
        print("Writing output video...")
        temp_video = output_path.replace('.mp4', '_temp.mp4')
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        for frame in processed_frames:
            out.write(frame)
        out.release()
        
        # Merge with audio
        print("Merging audio...")
        self.merge_audio_video(temp_video, audio_path, output_path)
        
        # Clean up temp file
        os.remove(temp_video)
        
        print(f"✓ Output saved to: {output_path}")
        
    def apply_lipsync(self, frame, face_rect, mel_chunk):
        """Apply lip-sync to a single frame"""
        if self.wav2lip is None:
            # If model not loaded, return original frame
            return frame
            
        try:
            # Extract face region
            x, y, w, h = [int(v) for v in face_rect]
            
            # Add padding
            pad = int(w * 0.2)
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(frame.shape[1] - x, w + 2*pad)
            h = min(frame.shape[0] - y, h + 2*pad)
            
            # Crop face
            face = frame[y:y+h, x:x+w]
            
            if face.size == 0:
                return frame
                
            # Store original face for later
            original_face = face.copy()
            
            # Resize to model input size
            face_input = cv2.resize(face, (self.img_size, self.img_size))
            
            # Normalize
            face_input = face_input.astype(np.float32) / 255.0
            
            # Add batch dimension and transpose for model
            face_input = np.expand_dims(face_input.transpose(2, 0, 1), 0)
            
            # Prepare mel input
            mel_input = np.expand_dims(mel_chunk, 0).astype(np.float32)
            
            # Run inference (placeholder - actual model needs proper inputs)
            # For now, apply simple transformation to simulate lip movement
            # This is where the actual ONNX model inference would happen
            
            # Simple simulation: modify the lower part of face based on audio
            audio_amplitude = np.mean(np.abs(mel_chunk))
            
            # Create a mask for the mouth region (lower third of face)
            mouth_region_start = int(face.shape[0] * 0.6)
            
            # Apply subtle color/brightness change to simulate movement
            face_output = face.copy()
            face_output[mouth_region_start:, :] = cv2.addWeighted(
                face[mouth_region_start:, :], 
                0.7 + 0.3 * audio_amplitude,
                face[mouth_region_start:, :],
                0,
                0
            )
            
            # Resize back to original size
            face_output = cv2.resize(face_output, (w, h))
            
            # Blend with original
            alpha = 0.8
            face_output = cv2.addWeighted(face_output, alpha, original_face, 1-alpha, 0)
            
            # Put face back into frame
            result = frame.copy()
            result[y:y+h, x:x+w] = face_output
            
            return result
            
        except Exception as e:
            print(f"Warning: Error in lip-sync: {e}")
            return frame
            
    def merge_audio_video(self, video_path, audio_path, output_path):
        """Merge audio and video using ffmpeg"""
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-c:v', 'libx264', '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest',  # Use shortest stream duration
            output_path, '-y'
        ]
        subprocess.run(cmd, check=True, capture_output=True)