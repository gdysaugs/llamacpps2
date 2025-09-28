import os
import cv2
import numpy as np
import onnxruntime as ort
import librosa
from tqdm import tqdm
import subprocess
import tempfile
from pathlib import Path

class Wav2LipFixed:
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
        
        # wav2lip-onnx-HQ parameters (from hparams.py)
        self.img_size = 96
        self.mel_step_size = 16
        self.fps = 25
        self.batch_size = 16
        
        # Audio parameters from wav2lip-onnx-HQ
        self.num_mels = 80
        self.n_fft = 800
        self.hop_size = 200
        self.win_size = 800
        self.sample_rate = 16000
        self.preemphasize = True
        self.preemphasis = 0.97
        self.min_level_db = -100
        self.ref_level_db = 20
        self.fmin = 55
        self.fmax = 7600
        
        # Normalization parameters
        self.signal_normalization = True
        self.symmetric_mels = True
        self.max_abs_value = 4.0
        self.allow_clipping_in_normalization = True
        
        self.load_models()
    
    def load_models(self):
        """Load the wav2lip ONNX model"""
        print("Loading Wav2Lip model...")
        
        wav2lip_path = os.path.join(self.model_path, 'wav2lip', 'wav2lip.onnx')
        if not os.path.exists(wav2lip_path):
            raise FileNotFoundError(f"Wav2Lip model not found: {wav2lip_path}")
        
        self.wav2lip_model = ort.InferenceSession(wav2lip_path, providers=self.providers)
        print(f"✓ Wav2Lip model loaded with providers: {self.wav2lip_model.get_providers()}")
        
        # Load face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def preemphasize_audio(self, wav):
        """Apply pre-emphasis filter"""
        if self.preemphasize:
            return np.append(wav[0], wav[1:] - self.preemphasis * wav[:-1])
        return wav
    
    def _normalize_mel(self, S):
        """Normalize mel spectrogram using wav2lip-onnx-HQ settings"""
        if self.signal_normalization:
            if self.symmetric_mels:
                # Symmetric normalization around 0
                S = ((S - self.min_level_db) / (-self.min_level_db)) * 2 - 1
                if self.allow_clipping_in_normalization:
                    S = np.clip(S, -self.max_abs_value, self.max_abs_value)
                S = S / self.max_abs_value
            else:
                # Asymmetric normalization [0, max_abs_value]
                S = (S - self.min_level_db) / (-self.min_level_db)
                if self.allow_clipping_in_normalization:
                    S = np.clip(S, 0, self.max_abs_value)
                S = S / self.max_abs_value
        return S
    
    def get_mel_spectrogram(self, audio_path):
        """Extract mel spectrogram using wav2lip-onnx-HQ method"""
        # Load audio at correct sample rate
        wav, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Apply pre-emphasis
        wav = self.preemphasize_audio(wav)
        
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            n_mels=self.num_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log mel and apply reference level
        mel_db = librosa.power_to_db(mel, ref=np.max) - self.ref_level_db
        
        # Apply wav2lip-onnx-HQ normalization
        mel_normalized = self._normalize_mel(mel_db)
        
        return mel_normalized
    
    def get_mel_chunks(self, mel, fps=25):
        """Create mel chunks synchronized with video frames"""
        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        
        i = 0
        while i < int(len(mel[0]) / mel_idx_multiplier):
            start_idx = int(i * mel_idx_multiplier)
            
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunk = mel[:, -self.mel_step_size:]
            else:
                mel_chunk = mel[:, start_idx:start_idx + self.mel_step_size]
            
            # Ensure correct shape: (80, 16)
            if mel_chunk.shape[1] < self.mel_step_size:
                padding = self.mel_step_size - mel_chunk.shape[1]
                mel_chunk = np.pad(mel_chunk, ((0, 0), (0, padding)), mode='edge')
            
            mel_chunks.append(mel_chunk)
            i += 1
        
        return mel_chunks
    
    def detect_faces(self, frames):
        """Detect faces in frames"""
        face_rects = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            
            if len(faces) > 0:
                # Get largest face
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                
                # Expand and square the face region
                padding = int(max(w, h) * 0.3)
                size = max(w, h) + padding
                
                x_center = x + w // 2
                y_center = y + h // 2
                
                x_new = max(0, x_center - size // 2)
                y_new = max(0, y_center - size // 2)
                
                face_rects.append([x_new, y_new, size, size])
            else:
                # Default center region
                h, w = frame.shape[:2]
                face_rects.append([w//4, h//4, w//2, h//2])
        
        return face_rects
    
    def crop_and_resize_face(self, frame, face_rect):
        """Crop and resize face to model input size"""
        x, y, w, h = [int(v) for v in face_rect]
        
        # Ensure we don't go out of bounds
        x_end = min(frame.shape[1], x + w)
        y_end = min(frame.shape[0], y + h)
        x = max(0, x)
        y = max(0, y)
        
        # Crop face
        face = frame[y:y_end, x:x_end]
        
        if face.size == 0:
            return None, None
        
        # Resize to model input size (96x96)
        face_resized = cv2.resize(face, (self.img_size, self.img_size))
        
        return face_resized, (x, y, x_end - x, y_end - y)
    
    def preprocess_frames_wav2lip_style(self, faces):
        """Preprocess faces using wav2lip-onnx-HQ method"""
        batch_size = len(faces)
        
        # Convert faces to numpy array and normalize to [0, 1]
        img_batch = np.array(faces, dtype=np.float32) / 255.0
        
        # Create 6-channel input: masked version + original
        # Mask the right half of the first 3 channels
        img_masked = img_batch.copy()
        img_masked[:, :, self.img_size//2:] = 0
        
        # Concatenate masked and original images along channel dimension
        # Shape: (batch, height, width, 6)
        img_batch_6ch = np.concatenate((img_masked, img_batch), axis=3)
        
        # Transpose to NCHW format: (batch, channels, height, width)
        img_batch_6ch = img_batch_6ch.transpose((0, 3, 1, 2)).astype(np.float32)
        
        return img_batch_6ch
    
    def preprocess_mel_wav2lip_style(self, mel_chunks):
        """Preprocess mel chunks using wav2lip-onnx-HQ method"""
        # Convert to numpy array
        mel_batch = np.array(mel_chunks)
        
        # Add channel dimension: (batch, 80, 16) -> (batch, 80, 16, 1)
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        
        # Transpose to NCHW format: (batch, 1, 80, 16)
        mel_batch = mel_batch.transpose((0, 3, 1, 2)).astype(np.float32)
        
        return mel_batch
    
    def postprocess_output(self, output):
        """Postprocess model output back to image"""
        # Model output should be in [0, 1] range, convert to [0, 255]
        # Transpose from CHW to HWC
        output = np.transpose(output, (1, 2, 0))
        
        # Convert to [0, 255] range
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        
        return output
    
    def inference(self, video_path, audio_path, output_path):
        """Main inference pipeline using wav2lip-onnx-HQ method"""
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
        
        print(f"Loaded {len(frames)} frames at {fps} FPS")
        
        # Process audio
        print("Processing audio...")
        mel = self.get_mel_spectrogram(audio_path)
        mel_chunks = self.get_mel_chunks(mel, fps)
        
        # Ensure mel chunks match frame count
        if len(mel_chunks) < len(frames):
            last_mel = mel_chunks[-1] if mel_chunks else np.zeros((80, 16))
            while len(mel_chunks) < len(frames):
                mel_chunks.append(last_mel)
        elif len(mel_chunks) > len(frames):
            mel_chunks = mel_chunks[:len(frames)]
        
        print(f"Generated {len(mel_chunks)} mel chunks")
        
        # Detect faces
        print("Detecting faces...")
        face_rects = self.detect_faces(frames)
        
        # Process frames
        print("Generating lip-sync...")
        output_frames = []
        
        for i in tqdm(range(0, len(frames), self.batch_size)):
            # Prepare batch
            batch_frames = frames[i:i + self.batch_size]
            batch_rects = face_rects[i:i + self.batch_size]
            batch_mels = mel_chunks[i:i + self.batch_size]
            
            # Crop faces
            batch_faces = []
            batch_coords = []
            
            for frame, rect in zip(batch_frames, batch_rects):
                face, coords = self.crop_and_resize_face(frame, rect)
                if face is not None:
                    batch_faces.append(face)
                    batch_coords.append(coords)
                else:
                    # Use center crop if face detection failed
                    h, w = frame.shape[:2]
                    center_crop = frame[h//4:3*h//4, w//4:3*w//4]
                    center_crop = cv2.resize(center_crop, (self.img_size, self.img_size))
                    batch_faces.append(center_crop)
                    batch_coords.append((w//4, h//4, w//2, h//2))
            
            if not batch_faces:
                continue
            
            # Preprocess using wav2lip-onnx-HQ method
            video_input = self.preprocess_frames_wav2lip_style(batch_faces)
            mel_input = self.preprocess_mel_wav2lip_style(batch_mels)
            
            # Run inference
            try:
                outputs = self.wav2lip_model.run(
                    None, 
                    {
                        'video_frames': video_input,
                        'mel_spectrogram': mel_input
                    }
                )
                
                predicted_frames = outputs[0]  # Shape: (batch, 3, 96, 96)
                
                # Postprocess and place back into original frames
                for j, (orig_frame, coords, pred_frame) in enumerate(zip(batch_frames, batch_coords, predicted_frames)):
                    # Postprocess prediction
                    lip_sync_face = self.postprocess_output(pred_frame)
                    
                    # Resize back and place into original frame
                    x, y, w, h = coords
                    lip_sync_face_resized = cv2.resize(lip_sync_face, (w, h))
                    
                    result_frame = orig_frame.copy()
                    result_frame[y:y+h, x:x+w] = lip_sync_face_resized
                    
                    output_frames.append(result_frame)
                    
            except Exception as e:
                print(f"Inference error: {e}")
                # Fallback to original frames
                output_frames.extend(batch_frames)
        
        # Write output video
        print("Writing output video...")
        self.write_video_with_audio(output_frames, audio_path, output_path, fps)
        print(f"✓ Output saved: {output_path}")
    
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
            os.rename(temp_video, output_path)