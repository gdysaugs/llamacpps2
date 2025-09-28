import os
import cv2
import numpy as np
import onnxruntime as ort
import librosa
from tqdm import tqdm
import subprocess
import tempfile
from pathlib import Path

class Wav2LipCorrect:
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
        
        # Wav2Lip parameters
        self.img_size = 96
        self.mel_step_size = 16
        self.fps = 25
        self.batch_size = 32
        
        self.load_models()
    
    def load_models(self):
        """Load the correct Wav2Lip ONNX model"""
        print("Loading Wav2Lip model...")
        
        # Load main wav2lip model
        wav2lip_path = os.path.join(self.model_path, 'wav2lip', 'wav2lip.onnx')
        if not os.path.exists(wav2lip_path):
            raise FileNotFoundError(f"Wav2Lip model not found: {wav2lip_path}")
        
        self.wav2lip_model = ort.InferenceSession(wav2lip_path, providers=self.providers)
        print(f"✓ Wav2Lip model loaded with providers: {self.wav2lip_model.get_providers()}")
        
        # Load face detection (simple OpenCV for now)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def get_mel_spectrogram(self, audio_path):
        """Extract mel spectrogram from audio"""
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
        
        # Normalize
        mel = (mel + 100) / 100
        mel = np.clip(mel, -1, 1)
        
        return mel
    
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
                # Pad if needed
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
                
                # Expand face region
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
    
    def preprocess_frames(self, faces):
        """Preprocess faces for model input"""
        # Normalize to [-1, 1] and convert to CHW format
        processed_faces = []
        
        for face in faces:
            # Normalize
            face_norm = face.astype(np.float32) / 127.5 - 1.0
            # Convert BGR to RGB and transpose to CHW
            face_rgb = cv2.cvtColor(face_norm, cv2.COLOR_BGR2RGB)
            face_chw = np.transpose(face_rgb, (2, 0, 1))
            processed_faces.append(face_chw)
        
        return np.array(processed_faces)
    
    def postprocess_output(self, output):
        """Postprocess model output back to image"""
        # Convert from CHW to HWC
        output = np.transpose(output, (1, 2, 0))
        
        # Check output range and normalize accordingly
        if output.max() <= 1.0:
            # Output is in [0, 1] range
            output = output * 255.0
        else:
            # Output might be in [-1, 1] range
            output = (output + 1.0) * 127.5
        
        # Clip and convert to uint8
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        # Convert from RGB to BGR if needed
        if output.shape[2] == 3:
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output
    
    def inference(self, video_path, audio_path, output_path):
        """Main inference pipeline"""
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
                    # Use original frame if face detection failed
                    batch_faces.append(cv2.resize(frame, (self.img_size, self.img_size)))
                    batch_coords.append((0, 0, width, height))
            
            if not batch_faces:
                continue
            
            # Preprocess
            video_input = self.preprocess_frames(batch_faces)
            mel_input = np.array(batch_mels)
            
            # Reshape for model: mel (batch, 1, 80, 16), video (batch, 6, 96, 96)
            mel_input = np.expand_dims(mel_input, axis=1)  # Add channel dimension
            
            # For video input, we need 6 channels (previous frames + current)
            # For simplicity, repeat current frame 6 times
            video_input_6ch = np.repeat(video_input, 2, axis=1)  # 3->6 channels
            
            # Run inference
            try:
                outputs = self.wav2lip_model.run(
                    None, 
                    {
                        'mel_spectrogram': mel_input.astype(np.float32),
                        'video_frames': video_input_6ch.astype(np.float32)
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