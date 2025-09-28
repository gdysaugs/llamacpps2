import os
import cv2
import numpy as np
import onnxruntime as ort
import librosa
from tqdm import tqdm
import subprocess
import audio

class Wav2LipFinal:
    def __init__(self, model_path, device='cuda', use_gfpgan=False, gfpgan_blend=0.5, use_gan_model=True, use_384_model=False, resize_factor=1):
        self.device = device
        self.model_path = model_path
        self.use_gfpgan = use_gfpgan
        self.gfpgan_blend = gfpgan_blend
        self.use_gan_model = use_gan_model
        self.use_384_model = use_384_model
        self.resize_factor = resize_factor
        
        # Setup CUDA
        if device == 'cuda':
            import torch
            if torch.cuda.is_available():
                cuda_path = os.path.dirname(torch.__file__) + "/lib"
                if cuda_path not in os.environ.get('LD_LIBRARY_PATH', ''):
                    os.environ['LD_LIBRARY_PATH'] = cuda_path + ":" + os.environ.get('LD_LIBRARY_PATH', '')
                
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                self.providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
                print(f"✓ Using GPU (CUDA) acceleration")
            else:
                self.providers = ['CPUExecutionProvider']
                self.device = 'cpu'
        else:
            self.providers = ['CPUExecutionProvider']
        
        # Model parameters - native resolution
        self.img_size = 96
        self.upscale_factor = 1  # Native resolution
        
        self.mel_step_size = 16
        self.batch_size = 1
        
        self.load_models()
        
        # Load GFPGAN if requested
        if self.use_gfpgan:
            self.load_gfpgan()
    
    def load_models(self):
        """Load wav2lip ONNX model"""
        # Choose model based on resolution and GAN preference
        if self.use_384_model:
            model_name = 'wav2lip_384.onnx'
            self.img_size = 384
        else:
            model_name = 'wav2lip_gan.onnx' if self.use_gan_model else 'wav2lip.onnx'
            self.img_size = 96
            
        print(f"Loading Wav2Lip model: {model_name} (resolution: {self.img_size}x{self.img_size})...")
        
        wav2lip_path = os.path.join(self.model_path, 'wav2lip', model_name)
        if not os.path.exists(wav2lip_path):
            if self.use_384_model:
                print(f"Warning: {model_name} not found. Falling back to wav2lip_gan.onnx with upscaling...")
                self.use_384_model = False
                self.img_size = 96
                self.upscale_factor = 4
                model_name = 'wav2lip_gan.onnx'
                wav2lip_path = os.path.join(self.model_path, 'wav2lip', model_name)
            
            if not os.path.exists(wav2lip_path):
                raise FileNotFoundError(f"Wav2Lip model not found: {wav2lip_path}")
        
        # wav2lip-onnx-HQ と同じセッション設定
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.wav2lip_model = ort.InferenceSession(wav2lip_path, sess_options=session_options, providers=self.providers)
        print(f"✓ Wav2Lip model loaded with providers: {self.wav2lip_model.get_providers()}")
        
        # RetinaFace detection
        try:
            from utils.retinaface import RetinaFace
            from utils.face_alignment import get_cropped_head_256
            retinaface_model = os.path.join(self.model_path, 'retinaface', 'scrfd_2.5g_bnkps.onnx')
            if os.path.exists(retinaface_model):
                self.detector = RetinaFace(retinaface_model, provider=self.providers, session_options=session_options)
                print(f"✓ RetinaFace model loaded: {retinaface_model}")
                self.use_retinaface = True
            else:
                print(f"Warning: RetinaFace model not found: {retinaface_model}")
                print("Falling back to OpenCV face detection")
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.use_retinaface = False
        except ImportError:
            print("Warning: RetinaFace not available, using OpenCV face detection")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.use_retinaface = False
    
    def load_gfpgan(self):
        """Load GFPGAN enhancer"""
        print("Loading GFPGAN enhancer...")
        gfpgan_path = os.path.join(self.model_path, 'gfpgan', 'GFPGANv1.4.onnx')
        if not os.path.exists(gfpgan_path):
            print(f"Warning: GFPGAN model not found: {gfpgan_path}")
            self.use_gfpgan = False
            return
            
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.gfpgan_model = ort.InferenceSession(gfpgan_path, sess_options=session_options, providers=self.providers)
        print(f"✓ GFPGAN model loaded with providers: {self.gfpgan_model.get_providers()}")
    
    def enhance_face_gfpgan(self, face_image):
        """Enhance face using GFPGAN with smooth processing for small inputs"""
        if not self.use_gfpgan:
            return face_image
            
        try:
            original_size = face_image.shape[:2]
            
            # Direct high-quality resize to 512x512 for GFPGAN (no intermediate steps)
            face_512 = cv2.resize(face_image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert BGR to RGB and normalize properly
            face_rgb = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb.astype(np.float32) / 255.0
            face_normalized = (face_normalized - 0.5) / 0.5  # [-1, 1]
            
            # Prepare for model input
            face_input = np.transpose(face_normalized, (2, 0, 1))
            face_batch = np.expand_dims(face_input, axis=0)
            
            # Run GFPGAN inference
            enhanced = self.gfpgan_model.run(None, {'input': face_batch.astype(np.float32)})[0]
            
            # Post-process output
            enhanced = enhanced[0]  # Remove batch dimension
            enhanced = np.transpose(enhanced, (1, 2, 0))  # CHW to HWC
            
            # Carefully denormalize
            enhanced = np.clip((enhanced + 1.0) / 2.0, 0.0, 1.0)
            enhanced = (enhanced * 255.0).astype(np.uint8)
            
            # Convert back to BGR
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            
            # High-quality resize back to original size
            enhanced_resized = cv2.resize(enhanced_bgr, (original_size[1], original_size[0]), interpolation=cv2.INTER_LANCZOS4)
            
            return enhanced_resized
            
        except Exception as e:
            print(f"GFPGAN enhancement failed: {e}")
            return face_image
    
    def detect_faces(self, frames):
        """Face detection using RetinaFace or OpenCV fallback"""
        face_rects = []
        
        for frame in frames:
            if self.use_retinaface:
                # Use RetinaFace detection
                dets, landmarks = self.detector.detect(frame, threshold=0.8)
                
                if len(dets) > 0:
                    # Get best detection (highest confidence)
                    best_det = dets[0]  # Already sorted by confidence
                    x1, y1, x2, y2, confidence = best_det
                    
                    # Convert to x, y, w, h format
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    # Expand face region
                    padding = int(max(w, h) * 0.3)
                    size = max(w, h) + padding
                    
                    x_center = x + w // 2
                    y_center = y + h // 2
                    
                    x_new = max(0, x_center - size // 2)
                    y_new = max(0, y_center - size // 2)
                    
                    face_rects.append([x_new, y_new, size, size])
                else:
                    # Default center region if no face detected
                    h, w = frame.shape[:2]
                    face_rects.append([w//4, h//4, w//2, h//2])
            else:
                # Use OpenCV face detection (fallback)
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
        """Crop face using wav2lip-onnx-HQ method with 256x256 alignment"""
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
        
        # Step 1: Create 256x256 aligned face (wav2lip-onnx-HQ method)
        crop_face = cv2.resize(face_region, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        # Step 2: Extract mouth region like wav2lip-onnx-HQ
        # crop_face[65-(padY):241-(padY),62:194] with padY=0
        padY = 0
        sub_face = crop_face[65-padY:241-padY, 62:194]  # 176x132 mouth region
        
        # Step 3: Resize to model input size
        sub_face_resized = cv2.resize(sub_face, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        
        return sub_face_resized, (x, y, x_end - x, y_end - y), crop_face
    
    def datagen(self, frames, mel_chunks):
        """wav2lip-onnx-HQ と完全に同じ datagen 関数"""
        img_batch, mel_batch = [], []

        for i, m in enumerate(mel_chunks):
            idx = i % len(frames)
            
            img_batch.append(frames[idx])
            mel_batch.append(m)

            img_batch_np, mel_batch_np = np.asarray(img_batch), np.asarray(mel_batch)

            # wav2lip-onnx-HQ と同じマスク処理
            img_masked = img_batch_np.copy()
            img_masked[:, self.img_size//2:] = 0
            
            # 6チャンネル作成: マスクされた画像 + オリジナル画像
            img_batch_6ch = np.concatenate((img_masked, img_batch_np), axis=3) / 255.
            mel_batch_reshaped = np.reshape(mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1])

            # transpose to NCHW
            img_batch_6ch = img_batch_6ch.transpose((0, 3, 1, 2)).astype(np.float32)
            mel_batch_reshaped = mel_batch_reshaped.transpose((0, 3, 1, 2)).astype(np.float32)

            yield img_batch_6ch, mel_batch_reshaped
            img_batch, mel_batch = [], []
    
    def inference(self, video_path, audio_path, output_path):
        """wav2lip-onnx-HQ と同じ推論パイプライン"""
        print(f"Processing: {video_path} + {audio_path}")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate processing dimensions based on resize_factor
        proc_width = orig_width // self.resize_factor
        proc_height = orig_height // self.resize_factor
        
        print(f"Original resolution: {orig_width}x{orig_height}")
        if self.resize_factor > 1:
            print(f"Processing resolution: {proc_width}x{proc_height} (resize_factor: {self.resize_factor})")
        
        frames = []
        orig_frames = []  # Keep original frames for final output
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            orig_frames.append(frame)  # Store original resolution frame
            
            # Resize for processing if needed
            if self.resize_factor > 1:
                frame = cv2.resize(frame, (proc_width, proc_height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        
        print(f"Loaded {len(frames)} frames at {fps} FPS")
        
        # Process audio using wav2lip-onnx-HQ audio module
        print("Processing audio...")
        
        # Create temp directory and file path (EXE compatible)
        import tempfile
        from pathlib import Path
        
        # Ensure temp directory exists
        temp_dir = Path(__file__).parent.parent / 'temp'
        temp_dir.mkdir(exist_ok=True)
        temp_wav = temp_dir / 'temp.wav'
        
        # First convert to wav
        subprocess.run(['ffmpeg', '-y', '-i', audio_path, '-ac', '1', '-strict', '-2', str(temp_wav)], check=True)
        
        # Load and process with wav2lip-onnx-HQ audio functions
        wav = audio.load_wav(str(temp_wav), 16000)
        mel = audio.melspectrogram(wav)
        
        # Create mel chunks exactly like wav2lip-onnx-HQ
        mel_chunks = []
        mel_idx_multiplier = 80. / fps 
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
            i += 1
        
        print(f"Generated {len(mel_chunks)} mel chunks")
        
        # Trim frames to match mel chunks
        frames = frames[:len(mel_chunks)]
        
        # Detect faces
        print("Detecting faces...")
        face_rects = self.detect_faces(frames)
        
        # Crop faces using wav2lip-onnx-HQ method
        sub_faces = []
        coords_list = []
        crop_faces = []  # Store 256x256 aligned faces
        
        for frame, rect in zip(frames, face_rects):
            face, coords, crop_face = self.crop_and_resize_face(frame, rect)
            if face is not None:
                sub_faces.append(face)
                coords_list.append(coords)
                crop_faces.append(crop_face)
            else:
                # Use center crop if face detection failed
                h, w = frame.shape[:2]
                center_region = frame[h//4:3*h//4, w//4:3*w//4]
                
                # Create 256x256 aligned face
                crop_face = cv2.resize(center_region, (256, 256), interpolation=cv2.INTER_CUBIC)
                
                # Extract mouth region
                padY = 0
                sub_face = crop_face[65-padY:241-padY, 62:194]
                sub_face_resized = cv2.resize(sub_face, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
                
                sub_faces.append(sub_face_resized)
                coords_list.append((w//4, h//4, w//2, h//2))
                crop_faces.append(crop_face)
        
        # Process using datagen
        print("Generating lip-sync...")
        gen = self.datagen(sub_faces, mel_chunks)
        
        output_frames = []
        
        for i, (img_batch, mel_batch) in enumerate(tqdm(gen, total=len(mel_chunks))):
            
            # Run wav2lip inference exactly like wav2lip-onnx-HQ
            pred = self.wav2lip_model.run(None, {
                'mel_spectrogram': mel_batch, 
                'video_frames': img_batch
            })[0][0]
            
            # Process output with proper clipping and color handling
            pred = pred.transpose(1, 2, 0)  # CHW to HWC
            pred = np.clip(pred, 0.0, 1.0)  # Ensure values are in [0, 1] range
            pred = (pred * 255.0).astype(np.uint8)  # Convert to [0, 255]
            
            # wav2lip-onnx-HQ method: Place prediction back into 256x256 aligned face
            orig_frame = frames[i]
            x, y, w, h = coords_list[i]
            crop_face = crop_faces[i].copy()
            
            # Step 1: Resize prediction back to mouth region size (176x132)
            pred_resized = cv2.resize(pred, (132, 176), interpolation=cv2.INTER_CUBIC)
            
            # Step 2: Place prediction into 256x256 aligned face (wav2lip-onnx-HQ coordinates)
            padY = 0
            crop_face[65-padY:241-padY, 62:194] = pred_resized
            
            # Step 3: Apply GFPGAN to entire 256x256 aligned face if enabled
            if self.use_gfpgan:
                enhanced_crop = self.enhance_face_gfpgan(crop_face)
                # Blend enhanced with original using gfpgan_blend ratio (weaker enhancement)
                crop_face = cv2.addWeighted(
                    enhanced_crop.astype(np.float32), self.gfpgan_blend,
                    crop_face.astype(np.float32), 1.0 - self.gfpgan_blend,
                    0.0
                ).astype(np.uint8)
            
            # Step 4: Resize 256x256 aligned face back to original face size
            face_final = cv2.resize(crop_face, (w, h), interpolation=cv2.INTER_CUBIC)
            
            # Step 5: Place back into original frame
            result_frame = orig_frame.copy()
            result_frame[y:y+h, x:x+w] = face_final
            
            # Scale result back to original resolution if needed
            if self.resize_factor > 1:
                result_frame = cv2.resize(result_frame, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
            
            output_frames.append(result_frame)
        
        # Write output video
        print("Writing output video...")
        self.write_video_with_audio(output_frames, audio_path, output_path, fps)
        print(f"✓ Output saved: {output_path}")
    
    def write_video_with_audio(self, frames, audio_path, output_path, fps):
        """Write video frames and merge with audio"""
        temp_video = output_path.replace('.mp4', '_temp.mp4')
        
        # Upscale to 1080p if needed
        target_height = 1080
        first_frame = frames[0]
        current_height, current_width = first_frame.shape[:2]
        
        if current_height != target_height:
            # Calculate target width maintaining aspect ratio
            aspect_ratio = current_width / current_height
            target_width = int(target_height * aspect_ratio)
            
            print(f"Upscaling to 1080p: {current_width}x{current_height} -> {target_width}x{target_height}")
            
            # Upscale all frames to 1080p
            upscaled_frames = []
            for frame in frames:
                upscaled = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
                upscaled_frames.append(upscaled)
            frames = upscaled_frames
            
            width, height = target_width, target_height
        else:
            height, width = current_height, current_width
        
        # Write video frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Merge with audio using wav2lip-onnx-HQ high quality settings
        cmd = [
            'ffmpeg', '-i', temp_video, '-i', audio_path,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '5',  # Highest quality (wav2lip-onnx-HQ setting)
            '-preset', 'slow',  # Best compression (wav2lip-onnx-HQ setting)
            '-c:a', 'libmp3lame', '-ac', '2', '-ar', '44100', '-ab', '128000',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', output_path, '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_video)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            os.rename(temp_video, output_path)