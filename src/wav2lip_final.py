import os
import cv2
import numpy as np
import onnxruntime as ort
import librosa
from tqdm import tqdm
import subprocess
import audio

class Wav2LipFinal:
    def __init__(self, model_path, device='cuda', use_gfpgan=False, gfpgan_blend=0.5, use_gan_model=True, use_384_model=False, resize_factor=1, gfpgan_mouth_only=False):
        self.device = device
        self.model_path = model_path
        self.use_gfpgan = use_gfpgan
        self.gfpgan_blend = gfpgan_blend
        self.gfpgan_mouth_only = gfpgan_mouth_only
        self.use_gan_model = use_gan_model
        self.use_384_model = use_384_model
        self.resize_factor = resize_factor
        self.initial_use_gfpgan = use_gfpgan  # Store initial setting for reload
        
        # Setup CUDA
        if device == 'cuda':
            import torch
            if torch.cuda.is_available():
                cuda_path = os.path.dirname(torch.__file__) + "/lib"
                if cuda_path not in os.environ.get('LD_LIBRARY_PATH', ''):
                    os.environ['LD_LIBRARY_PATH'] = cuda_path + ":" + os.environ.get('LD_LIBRARY_PATH', '')

                # GPU optimization environment variables
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
                os.environ['CUDA_MODULE_LOADING'] = 'EAGER'
                # Optimized CUDA provider configuration for CUDA 12
                cuda_provider_options = {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 3 * 1024 * 1024 * 1024,  # 3GB limit for RTX 3050
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                    "cudnn_conv_use_max_workspace": True
                }
                self.providers = [("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"]
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
        
        self.batch_size = 4
        
        self.load_models()
        
        # Load GFPGAN if requested
        if self.use_gfpgan:
            self.load_gfpgan()
    
    def load_models(self):
        # ONNX Session optimization
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        if self.device == 'cuda':
            session_options.intra_op_num_threads = 2  # GPU用に軽量設定
        else:
            session_options.intra_op_num_threads = 4  # CPU並列化
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
        
        self.wav2lip_model = ort.InferenceSession(wav2lip_path, sess_options=session_options, providers=self.providers)
        print(f"✓ Wav2Lip model loaded with providers: {self.wav2lip_model.get_providers()}")
        
        # RetinaFace detection (PyTorch)
        try:
            from utils.retinaface_pytorch import RetinaFaceDetector
            from utils.face_alignment import get_cropped_head_256
            retinaface_model = os.path.join(self.model_path, 'detection_Resnet50_Final.pth')
            if os.path.exists(retinaface_model):
                self.detector = RetinaFaceDetector(retinaface_model, device=self.device)
                print(f"✓ PyTorch RetinaFace model loaded: {retinaface_model}")
                self.use_retinaface = True
            else:
                print(f"Warning: PyTorch RetinaFace model not found: {retinaface_model}")
                print("Falling back to OpenCV face detection")
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.use_retinaface = False
        except ImportError as e:
            print(f"Warning: PyTorch RetinaFace not available: {e}")
            print("Using OpenCV face detection")
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
        """Enhanced GFPGAN with FP16 inference for speed"""
        if not self.use_gfpgan:
            return face_image

        try:
            original_size = face_image.shape[:2]

            # Direct high-quality resize to 512x512 for GFPGAN
            face_512 = cv2.resize(face_image, (512, 512), interpolation=cv2.INTER_LANCZOS4)

            # Convert BGR to RGB and normalize properly
            face_rgb = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb.astype(np.float32) / 255.0
            face_normalized = (face_normalized - 0.5) / 0.5  # [-1, 1]

            # Prepare for model input
            face_input = np.transpose(face_normalized, (2, 0, 1))
            face_batch = np.expand_dims(face_input, axis=0)

            # Try FP16 inference for speed (fallback to FP32)
            try:
                # Convert to FP16 for faster inference
                face_batch_fp16 = face_batch.astype(np.float16)
                enhanced = self.gfpgan_model.run(None, {'input': face_batch_fp16})[0]
            except:
                # Fallback to FP32 if FP16 fails
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

    def enhance_faces_gfpgan_batch(self, face_images):
        """Batch GFPGAN processing for improved speed"""
        if not self.use_gfpgan or len(face_images) == 0:
            return face_images

        try:
            original_sizes = [img.shape[:2] for img in face_images]
            batch_size = min(4, len(face_images))  # Max batch size 4
            enhanced_faces = []

            # Process in batches of 4
            for i in range(0, len(face_images), batch_size):
                batch_end = min(i + batch_size, len(face_images))
                batch_faces = face_images[i:batch_end]
                batch_original_sizes = original_sizes[i:batch_end]

                # Prepare batch input
                batch_inputs = []
                for face_image in batch_faces:
                    # Resize to 512x512
                    face_512 = cv2.resize(face_image, (512, 512), interpolation=cv2.INTER_LANCZOS4)

                    # Convert and normalize
                    face_rgb = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB)
                    face_normalized = face_rgb.astype(np.float32) / 255.0
                    face_normalized = (face_normalized - 0.5) / 0.5  # [-1, 1]

                    # Prepare for model input
                    face_input = np.transpose(face_normalized, (2, 0, 1))
                    batch_inputs.append(face_input)

                # Stack into batch
                face_batch = np.stack(batch_inputs, axis=0)

                # Try FP16 batch inference
                try:
                    face_batch_fp16 = face_batch.astype(np.float16)
                    enhanced_batch = self.gfpgan_model.run(None, {'input': face_batch_fp16})[0]
                except:
                    enhanced_batch = self.gfpgan_model.run(None, {'input': face_batch.astype(np.float32)})[0]

                # Process batch output
                for j, (enhanced, original_size) in enumerate(zip(enhanced_batch, batch_original_sizes)):
                    # Post-process
                    enhanced = np.transpose(enhanced, (1, 2, 0))  # CHW to HWC
                    enhanced = np.clip((enhanced + 1.0) / 2.0, 0.0, 1.0)
                    enhanced = (enhanced * 255.0).astype(np.uint8)

                    # Convert back to BGR
                    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

                    # Resize back to original size
                    enhanced_resized = cv2.resize(enhanced_bgr, (original_size[1], original_size[0]), interpolation=cv2.INTER_LANCZOS4)
                    enhanced_faces.append(enhanced_resized)

            return enhanced_faces

        except Exception as e:
            print(f"GFPGAN batch enhancement failed: {e}")
            return face_images

    def enhance_faces_gfpgan_batch(self, face_images):
        """Batch GFPGAN processing for improved speed"""
        if not self.use_gfpgan or len(face_images) == 0:
            return face_images

        try:
            original_sizes = [img.shape[:2] for img in face_images]

            # Prepare batch input
            batch_inputs = []
            for face_image in face_images:
                # Resize to 512x512
                face_512 = cv2.resize(face_image, (512, 512), interpolation=cv2.INTER_LANCZOS4)

                # Convert and normalize
                face_rgb = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB)
                face_normalized = face_rgb.astype(np.float32) / 255.0
                face_normalized = (face_normalized - 0.5) / 0.5  # [-1, 1]

                # Prepare for model input
                face_input = np.transpose(face_normalized, (2, 0, 1))
                batch_inputs.append(face_input)

            # Stack into batch
            face_batch = np.stack(batch_inputs, axis=0)

            # Try FP16 batch inference
            try:
                face_batch_fp16 = face_batch.astype(np.float16)
                enhanced_batch = self.gfpgan_model.run(None, {'input': face_batch_fp16})[0]
            except:
                enhanced_batch = self.gfpgan_model.run(None, {'input': face_batch.astype(np.float32)})[0]

            # Process batch output
            enhanced_faces = []
            for i, (enhanced, original_size) in enumerate(zip(enhanced_batch, original_sizes)):
                # Post-process
                enhanced = np.transpose(enhanced, (1, 2, 0))  # CHW to HWC
                enhanced = np.clip((enhanced + 1.0) / 2.0, 0.0, 1.0)
                enhanced = (enhanced * 255.0).astype(np.uint8)

                # Convert back to BGR
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

                # Resize back to original size
                enhanced_resized = cv2.resize(enhanced_bgr, (original_size[1], original_size[0]), interpolation=cv2.INTER_LANCZOS4)
                enhanced_faces.append(enhanced_resized)

            return enhanced_faces

        except Exception as e:
            print(f"GFPGAN batch enhancement failed: {e}")
            return face_images
    
    def debug_face_detection(self, frame, frame_idx=0):
        """可視化デバッグ用：顔検出結果を画像として保存"""
        debug_frame = frame.copy()
        h, w = frame.shape[:2]

        if self.use_retinaface:
            try:
                # RetinaFace detection (lowered threshold for better detection)
                dets, landmarks = self.detector.detect(frame, threshold=0.3)
                print(f"🔍 Frame {frame_idx}: Detections={len(dets)}, Landmarks={len(landmarks)}")

                if len(dets) > 0 and len(landmarks) > 0:
                    for i, (det, landmark) in enumerate(zip(dets, landmarks)):
                        x1, y1, x2, y2, confidence = det
                        print(f"  Face {i}: confidence={confidence:.3f}, box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

                        # Draw face box
                        cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(debug_frame, f'Face {i}: {confidence:.2f}',
                                  (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Draw landmarks
                        landmark_points = landmark.reshape(-1, 2)
                        for j, (lx, ly) in enumerate(landmark_points):
                            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][j]
                            cv2.circle(debug_frame, (int(lx), int(ly)), 3, color, -1)
                            cv2.putText(debug_frame, str(j), (int(lx+5), int(ly)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                else:
                    print(f"  No faces detected with RetinaFace")
            except Exception as e:
                print(f"  RetinaFace detection error: {e}")

        # Save debug image
        debug_path = f"debug_face_detection_{frame_idx:04d}.jpg"
        cv2.imwrite(debug_path, debug_frame)
        print(f"💾 Debug image saved: {debug_path}")
        return debug_path

    def detect_faces_with_landmarks(self, frames):
        """Face detection with landmarks using RetinaFace"""
        face_data = []

        for frame in frames:
            frame_data = {
                'face_rect': None,
                'landmarks': None,
                'face_found': False
            }

            if self.use_retinaface:
                try:
                    # Use RetinaFace detection with landmarks (lowered threshold for better detection)
                    dets, landmarks = self.detector.detect(frame, threshold=0.3)
                    print(f"Dets shape: {dets.shape if len(dets) > 0 else 'empty'}, Landmarks shape: {landmarks.shape if len(landmarks) > 0 else 'empty'}")

                    if len(dets) > 0 and len(landmarks) > 0:
                        # Get best detection (highest confidence)
                        best_det = dets[0]  # Already sorted by confidence
                        best_landmarks = landmarks[0]  # Corresponding landmarks

                        x1, y1, x2, y2, confidence = best_det

                        print(f"RetinaFace detected face: confidence={confidence:.3f}, box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                        print(f"Landmarks: {best_landmarks.reshape(-1, 2)}")

                        # Check if coordinates are valid
                        if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                            # Store face detection data
                            frame_data['face_rect'] = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                            frame_data['landmarks'] = best_landmarks.reshape(-1, 2)  # 5 points: left_eye, right_eye, nose, left_mouth, right_mouth
                            frame_data['face_found'] = True
                            print(f"Valid face rect: {frame_data['face_rect']}")
                        else:
                            print(f"Invalid face coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                except Exception as e:
                    print(f"RetinaFace detection failed: {e}")

            if not frame_data['face_found']:
                # Fallback - use OpenCV or center region
                if hasattr(self, 'face_cascade'):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

                    if len(faces) > 0:
                        # Get largest face
                        face = max(faces, key=lambda x: x[2] * x[3])
                        frame_data['face_rect'] = list(face)
                        frame_data['face_found'] = True
                        # No landmarks for OpenCV detection
                        frame_data['landmarks'] = None

                if not frame_data['face_found']:
                    # Improved default face region for portrait videos (center-upper area)
                    h, w = frame.shape[:2]
                    # For vertical videos, face is usually in upper-center area
                    face_w = min(w//2, h//3)  # Square-ish face region
                    face_h = face_w
                    face_x = (w - face_w) // 2  # Center horizontally
                    face_y = h // 4  # Upper part, not center
                    frame_data['face_rect'] = [face_x, face_y, face_w, face_h]
                    frame_data['landmarks'] = None
                    frame_data['face_found'] = True
                    print(f"  Fallback face region: [{face_x}, {face_y}, {face_w}, {face_h}] for frame {w}x{h}")

            face_data.append(frame_data)

        return face_data
    
    def align_face_with_landmarks(self, frame, landmarks):
        """Align face using landmarks based on wav2lip-onnx-HQ approach"""
        # FFHQ template landmarks for 512x512 (scaled down to 256x256)
        template_landmarks = np.array([
            [127.675, 105.225],  # left eye
            [128.325, 105.225],  # right eye
            [128.0, 120.5],      # nose
            [115.5, 140.5],      # left mouth corner
            [140.5, 140.5]       # right mouth corner
        ], dtype=np.float32)

        if landmarks is None or len(landmarks) < 5:
            # Fallback to simple cropping
            return self.crop_face_simple(frame, landmarks)

        # Calculate affine transformation
        try:
            # Use first 5 landmarks (eyes, nose, mouth corners)
            src_landmarks = landmarks[:5].astype(np.float32)

            # Calculate transformation matrix
            tform = cv2.estimateAffinePartial2D(src_landmarks, template_landmarks)[0]

            if tform is None:
                print("Failed to estimate transformation matrix, using simple crop")
                return self.crop_face_simple(frame, landmarks)

            # Apply transformation to align face to 256x256
            aligned_face = cv2.warpAffine(frame, tform, (256, 256), flags=cv2.INTER_CUBIC)

            # Extract mouth region using wav2lip-onnx-HQ coordinates
            # crop_face[65-(padY):248-(padY),62:194] with padY=0
            padY = 0
            sub_face = aligned_face[65-padY:248-padY, 62:194]  # 183x132 mouth region

            if sub_face.size == 0:
                print("Empty sub_face after alignment, using simple crop")
                return self.crop_face_simple(frame, landmarks)

            # Resize to model input size
            sub_face_resized = cv2.resize(sub_face, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

            # Store transformation matrix for inverse transform
            self.inverse_tform = cv2.invertAffineTransform(tform)

            return sub_face_resized, tform, aligned_face

        except Exception as e:
            print(f"Face alignment failed: {e}, using simple crop")
            return self.crop_face_simple(frame, landmarks)

    def crop_face_simple_with_rect(self, frame, x, y, w, h):
        """Simple face cropping using detected face rectangle"""
        frame_h, frame_w = frame.shape[:2]

        # Expand face region by 10% for minimal context (reduced from 20%)
        padding = int(max(w, h) * 0.1)

        # Calculate expansion bounds with proper boundary handling
        x_center = x + w // 2
        y_center = y + h // 2

        # Create square crop region
        crop_size = max(w, h) + padding
        half_crop = crop_size // 2

        # Calculate ideal crop bounds
        ideal_x1 = x_center - half_crop
        ideal_y1 = y_center - half_crop
        ideal_x2 = x_center + half_crop
        ideal_y2 = y_center + half_crop

        # Clamp to frame boundaries and adjust to maintain aspect ratio
        x1 = max(0, ideal_x1)
        y1 = max(0, ideal_y1)
        x2 = min(frame_w, ideal_x2)
        y2 = min(frame_h, ideal_y2)

        # Ensure we have a valid crop region
        actual_w = x2 - x1
        actual_h = y2 - y1

        if actual_w <= 0 or actual_h <= 0:
            # Fallback to original face region
            x1, y1 = x, y
            x2, y2 = x + w, y + h

        # Crop face region
        face_region = frame[y1:y2, x1:x2]

        if face_region.size == 0:
            print(f"Empty face region at {x1}, {y1}, {x2}, {y2}")
            return None, None, None

        # Resize to 256x256 with proper aspect ratio handling
        region_h, region_w = face_region.shape[:2]

        # Always use the original face region without forced square padding
        # This prevents artifacts from padding
        crop_face = cv2.resize(face_region, (256, 256), interpolation=cv2.INTER_CUBIC)

        # Extract mouth region using exact wav2lip-onnx-HQ coordinates
        padY = 0
        sub_face = crop_face[65-padY:248-padY, 62:194]  # 183x132 mouth region

        if sub_face.size == 0:
            print("Empty sub_face after mouth extraction")
            return None, None, None

        # Resize to model input size
        sub_face_resized = cv2.resize(sub_face, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        # Debug output
        print(f"Face crop: ({x1}, {y1}, {x2-x1}, {y2-y1}) -> crop_face: {crop_face.shape} -> sub_face: {sub_face.shape} -> resized: {sub_face_resized.shape}")

        # Return both original face coordinates and expanded crop coordinates
        crop_coords = (x1, y1, x2-x1, y2-y1)  # Expanded crop region
        face_coords = (x, y, w, h)  # Original face detection
        return sub_face_resized, (crop_coords, face_coords), crop_face

    def crop_face_simple(self, frame, landmarks):
        """Simple face cropping fallback method"""
        h, w = frame.shape[:2]

        if landmarks is not None and len(landmarks) >= 5:
            # Use landmarks to determine face center
            face_center = np.mean(landmarks, axis=0)
            cx, cy = int(face_center[0]), int(face_center[1])

            # Calculate face size based on eye distance
            eye_dist = np.linalg.norm(landmarks[0] - landmarks[1])
            face_size = int(eye_dist * 2.5)  # Approximate face size
        else:
            # Use center of frame - but reduce size for better face detection
            cx, cy = w // 2, h // 2
            face_size = min(w, h) // 2  # Larger face region for better quality

        # Create square crop
        half_size = face_size // 2
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)

        # Crop face region
        face_region = frame[y1:y2, x1:x2]

        if face_region.size == 0:
            # Emergency fallback
            face_region = frame[h//4:3*h//4, w//4:3*w//4]

        # Resize to 256x256 for consistent processing
        crop_face = cv2.resize(face_region, (256, 256), interpolation=cv2.INTER_CUBIC)

        # Extract mouth region
        padY = 0
        sub_face = crop_face[65-padY:248-padY, 62:194]

        if sub_face.size == 0:
            # Use center region if mouth extraction fails
            sub_face = crop_face[64:192, 64:192]  # 128x128 center region

        # Resize to model input size
        sub_face_resized = cv2.resize(sub_face, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        # Return both crop coordinates and face coordinates (same for fallback)
        crop_coords = (x1, y1, x2-x1, y2-y1)
        return sub_face_resized, (crop_coords, crop_coords), crop_face
    
    def datagen(self, frames, mel_chunks):
        """wav2lip-onnx-HQ exact datagen function"""
        img_batch, mel_batch, frame_batch = [], [], []

        for i, m in enumerate(mel_chunks):
            idx = i % len(frames)  # non-static mode
            
            frame_to_save = frames[idx].copy()
            frame_batch.append(frame_to_save)
            
            img_batch.append(frames[idx])
            mel_batch.append(m)

            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0
            
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch
            img_batch, mel_batch, frame_batch = [], [], []
    
    def inference(self, video_path, audio_path, output_path):
        """wav2lip-onnx-HQ と同じ推論パイプライン"""
        print(f"Processing: {video_path} + {audio_path}")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Original resolution: {orig_width}x{orig_height}")

        # Calculate processing dimensions based on resize_factor
        proc_width = orig_width // self.resize_factor
        proc_height = orig_height // self.resize_factor
        if self.resize_factor > 1:
            print(f"Processing resolution: {proc_width}x{proc_height} (resize_factor: {self.resize_factor})")
        
        frames = []
        orig_frames = []  # Keep original frames for final output
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            orig_frames.append(frame)  # Store original resolution frame

            # Apply resize_factor if needed
            if self.resize_factor > 1:
                frame = cv2.resize(frame, (proc_width, proc_height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        
        print(f"Loaded {len(frames)} frames at {fps} FPS")
        
        # Process audio using wav2lip-onnx-HQ audio module
        print("Processing audio...")
        
        # Create temp directory and file path (EXE compatible)
        import tempfile
        import uuid
        from pathlib import Path
        
        # Ensure temp directory exists
        temp_dir = Path(__file__).parent.parent / 'temp'
        temp_dir.mkdir(exist_ok=True)
        # ランダムな一時ファイル名を生成
        temp_wav = temp_dir / f'temp_{uuid.uuid4().hex[:8]}.wav'
        
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

        # Don't trim frames - let them loop to match audio length
        video_loops = (len(mel_chunks) // len(frames)) + 1
        if len(mel_chunks) > len(frames):
            print(f"Audio longer than video: will loop video {video_loops} times to match audio length")
        
        # Detect faces with landmarks
        print("Detecting faces with landmarks...")
        face_data = self.detect_faces_with_landmarks(frames)

        # Debug face detection
        valid_faces = sum(1 for data in face_data if data['face_found'])
        landmarks_detected = sum(1 for data in face_data if data['landmarks'] is not None)
        print(f"Face detection results: {valid_faces}/{len(face_data)} frames have valid faces")
        print(f"Landmark detection results: {landmarks_detected}/{len(face_data)} frames have landmarks")

        if valid_faces == 0:
            print("❌ WARNING: No faces detected in any frame! Lip-sync will not work.")
        elif valid_faces < len(face_data) * 0.5:
            print(f"⚠️ WARNING: Only {valid_faces}/{len(face_data)} frames have detected faces.")

        # Crop and align faces using landmarks
        sub_faces = []
        coords_list = []
        crop_faces = []  # Store 256x256 aligned faces
        alignment_data = []  # Store alignment info for reconstruction

        frame_count = 0
        for frame, data in zip(frames, face_data):
            alignment_info = {'use_landmarks': False, 'tform': None, 'landmarks': None}

            # Debug first few frames with visualization
            if frame_count < 3:
                print(f"Frame {frame_count}: face_found={data['face_found']}, face_rect={data['face_rect']}, has_landmarks={data['landmarks'] is not None}")
                # Save debug visualization for first 3 frames
                debug_path = self.debug_face_detection(frame, frame_count)
                print(f"🔍 Debug visualization saved: {debug_path}")
            elif data['face_found']:
                print(f"Frame {frame_count}: face_found={data['face_found']}, face_rect={data['face_rect']}, has_landmarks={data['landmarks'] is not None}")

            if data['face_found'] and data['landmarks'] is not None:
                # Use detected face rect with landmarks
                x, y, w, h = data['face_rect']
                print(f"Frame {frame_count}: Using detected face rect with landmarks: ({x}, {y}, {w}, {h})")
                face, coords, crop_face = self.crop_face_simple_with_rect(frame, x, y, w, h)
                alignment_info['landmarks'] = data['landmarks']
            elif data['face_found']:
                # Use simple cropping with face rect
                if data['face_rect'] is not None:
                    x, y, w, h = data['face_rect']
                    print(f"Frame {frame_count}: Using simple face rect: ({x}, {y}, {w}, {h})")
                    face, coords, crop_face = self.crop_face_simple_with_rect(frame, x, y, w, h)
                else:
                    print(f"Frame {frame_count}: Face found but no face_rect, using fallback")
                    face, coords, crop_face = self.crop_face_simple(frame, None)
            else:
                # Emergency fallback - center crop
                print(f"Frame {frame_count}: No face found, using center fallback")
                face, coords, crop_face = self.crop_face_simple(frame, None)

            frame_count += 1

            if face is not None:
                sub_faces.append(face)
                coords_list.append(coords if coords is not None else (0, 0, frame.shape[1], frame.shape[0]))
                crop_faces.append(crop_face)
                alignment_data.append(alignment_info)
            else:
                print(f"Failed to process frame, using emergency fallback")
                # Emergency fallback
                h, w = frame.shape[:2]
                center_region = frame[h//4:3*h//4, w//4:3*w//4]
                crop_face = cv2.resize(center_region, (256, 256), interpolation=cv2.INTER_CUBIC)

                # Extract mouth region
                padY = 0
                sub_face = crop_face[65-padY:248-padY, 62:194]
                if sub_face.size == 0:
                    sub_face = crop_face[64:192, 64:192]  # 128x128 center
                sub_face_resized = cv2.resize(sub_face, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

                sub_faces.append(sub_face_resized)
                coords_list.append((w//4, h//4, w//2, h//2))
                crop_faces.append(crop_face)
        
        # Debug face cropping
        print(f"Successfully cropped {len(sub_faces)} faces from {len(frames)} frames")
        print(f"Generated {len(mel_chunks)} mel chunks for audio")

        if len(sub_faces) == 0:
            print("❌ ERROR: No valid face crops extracted! Cannot generate lip-sync.")
            return

        # Process using datagen
        print("Generating lip-sync...")
        gen = self.datagen(sub_faces, mel_chunks)

        print(f"Starting Wav2Lip inference for {len(mel_chunks)} audio chunks...")
        
        output_frames = []
        
        for i, (img_batch, mel_batch, frames_batch) in enumerate(tqdm(gen, total=len(mel_chunks))):

            # wav2lip-onnx-HQ exact inference
            img_batch = img_batch.transpose((0, 3, 1, 2)).astype(np.float32)
            mel_batch = mel_batch.transpose((0, 3, 1, 2)).astype(np.float32)

            # Debug input shapes
            if i < 3:  # Debug first 3 iterations
                print(f"🔍 Batch {i}: mel_batch shape={mel_batch.shape}, img_batch shape={img_batch.shape}")

            pred = self.wav2lip_model.run(None, {
                'mel_spectrogram': mel_batch,
                'video_frames': img_batch
            })[0][0]

            # Debug prediction output
            if i < 3:
                print(f"🔍 Batch {i}: pred raw shape={pred.shape}, min={pred.min():.3f}, max={pred.max():.3f}")

            # wav2lip-onnx-HQ exact post-processing
            pred = pred.transpose(1, 2, 0) * 255
            pred = pred.astype(np.uint8)
            pred = pred.reshape((1, self.img_size, self.img_size, 3))

            # Debug processed prediction
            if i < 3:
                print(f"🔍 Batch {i}: pred processed shape={pred.shape}, min={pred.min()}, max={pred.max()}")
                # Save prediction image for debug
                cv2.imwrite(f"debug_prediction_{i:03d}.jpg", pred[0][:, :, ::-1])  # BGR for OpenCV

            # Process each prediction (normally just one)
            for p, f in zip(pred, frames_batch):
                # Audio-driven video looping: repeat video until audio ends
                frame_idx = i % len(frames)
                coord_idx = i % len(coords_list)
                align_idx = coord_idx % len(alignment_data)

                orig_frame = frames[frame_idx]
                # Unpack coordinate tuples (crop_coords, face_coords)
                coords = coords_list[coord_idx]
                if isinstance(coords[0], tuple):
                    crop_coords, face_coords = coords
                    cx, cy, cw, ch = crop_coords  # Expanded crop region for placement
                    x, y, w, h = face_coords      # Original face detection (for reference)
                else:
                    # Fallback for old format
                    cx, cy, cw, ch = coords
                    x, y, w, h = coords
                crop_face = crop_faces[coord_idx].copy()
                align_info = alignment_data[align_idx]

                # Debug frame composition
                if i < 3:
                    print(f"🔍 Batch {i}: orig_frame shape={orig_frame.shape}, crop_face shape={crop_face.shape}")
                    print(f"🔍 Batch {i}: crop_coords={crop_coords}, face_coords={face_coords}")

                # Place lip-sync prediction with high-quality upscaling
                padY = 0
                # Two-stage upscaling for better quality: 96x96 -> 120x120 -> 132x183 (extended bottom)
                p_intermediate = cv2.resize(p, (120, 120), interpolation=cv2.INTER_LANCZOS4)
                p_resized = cv2.resize(p_intermediate, (132, 183), interpolation=cv2.INTER_LANCZOS4)

                if i < 3:
                    print(f"🔍 Batch {i}: p shape before resize={p.shape}, after resize={p_resized.shape}")
                    # Save resized prediction for debug
                    cv2.imwrite(f"debug_prediction_resized_{i:03d}.jpg", p_resized[:, :, ::-1])  # BGR for OpenCV

                p = p_resized

                # Apply edge-preserving filter to reduce blur
                p = cv2.bilateralFilter(p, 5, 80, 80)

                # Create blending mask for smoother integration (extended bottom)
                mask = np.ones((183, 132), dtype=np.float32)

                # Create feathered edges (5 pixel blend zone for sharper result)
                edge_size = 5
                for j in range(edge_size):
                    alpha = (j + 1) / edge_size
                    # Top and bottom edges
                    mask[j, :] = alpha
                    mask[-(j+1), :] = alpha
                    # Left and right edges
                    mask[:, j] = np.minimum(mask[:, j], alpha)
                    mask[:, -(j+1)] = np.minimum(mask[:, -(j+1)], alpha)

                # Apply blended mouth region (extended bottom) with boundary checks
                y1, y2 = 65-padY, 248-padY
                x1, x2 = 62, 194

                # Ensure coordinates are within crop_face bounds
                crop_h, crop_w = crop_face.shape[:2]
                y1 = max(0, min(y1, crop_h-1))
                y2 = max(y1+1, min(y2, crop_h))
                x1 = max(0, min(x1, crop_w-1))
                x2 = max(x1+1, min(x2, crop_w))

                # Get actual dimensions after boundary check
                actual_h, actual_w = y2-y1, x2-x1

                if i < 3:
                    print(f"🔍 Batch {i}: mouth region coords=({x1},{y1},{x2},{y2}), actual_size=({actual_w},{actual_h})")

                # Resize p to match actual region size
                if p.shape[:2] != (actual_h, actual_w):
                    p_resized = cv2.resize(p, (actual_w, actual_h), interpolation=cv2.INTER_CUBIC)
                    # Also resize mask
                    mask_resized = cv2.resize(mask, (actual_w, actual_h), interpolation=cv2.INTER_CUBIC)
                    if i < 3:
                        print(f"🔍 Batch {i}: resized p from {p.shape} to {p_resized.shape}")
                else:
                    p_resized = p
                    mask_resized = mask

                original_region = crop_face[y1:y2, x1:x2].astype(np.float32)
                p_float = p_resized.astype(np.float32)

                if i < 3:
                    print(f"🔍 Batch {i}: original_region shape={original_region.shape}, p_resized shape={p_resized.shape}")
                    # Save original mouth region for debug
                    cv2.imwrite(f"debug_original_mouth_{i:03d}.jpg", original_region.astype(np.uint8)[:, :, ::-1])
                    # Save final prediction for mouth region
                    cv2.imwrite(f"debug_final_prediction_{i:03d}.jpg", p_resized[:, :, ::-1])

                # Blend with resized mask
                mask_3d = np.stack([mask_resized, mask_resized, mask_resized], axis=2)
                blended = original_region * (1 - mask_3d) + p_float * mask_3d
                crop_face[y1:y2, x1:x2] = blended.astype(np.uint8)

                if i < 3:
                    print(f"🔍 Batch {i}: blending complete, mask_3d shape={mask_3d.shape}")
                    # Save blended result
                    cv2.imwrite(f"debug_blended_mouth_{i:03d}.jpg", blended.astype(np.uint8)[:, :, ::-1])

                # Apply GFPGAN enhancement
                if self.use_gfpgan:
                    if self.gfpgan_mouth_only:
                        # Apply GFPGAN only to mouth region (extended bottom)
                        y1, y2 = 65-padY, 248-padY
                        x1, x2 = 62, 194
                        mouth_region = crop_face[y1:y2, x1:x2].copy()  # 183x132

                        # Enhance the mouth region
                        enhanced_mouth = self.enhance_face_gfpgan(mouth_region)

                        # Blend enhanced mouth with original
                        blended_mouth = cv2.addWeighted(
                            enhanced_mouth.astype(np.float32), self.gfpgan_blend,
                            mouth_region.astype(np.float32), 1.0 - self.gfpgan_blend,
                            0.0
                        ).astype(np.uint8)

                        # Replace mouth region in face
                        crop_face[y1:y2, x1:x2] = blended_mouth
                    else:
                        # Apply GFPGAN to entire face
                        original_face = crop_face.copy()
                        enhanced_face = self.enhance_face_gfpgan(crop_face)
                        crop_face = cv2.addWeighted(
                            enhanced_face.astype(np.float32), self.gfpgan_blend,
                            original_face.astype(np.float32), 1.0 - self.gfpgan_blend,
                            0.0
                        ).astype(np.uint8)

                # Simple direct placement method
                result_frame = orig_frame.copy()

                # Use expanded crop coordinates for proper placement
                if cw > 0 and ch > 0 and crop_face.size > 0:
                    try:
                        # Resize crop_face to the expanded region size
                        face_final = cv2.resize(crop_face, (cw, ch), interpolation=cv2.INTER_CUBIC)

                        # Ensure bounds are correct
                        frame_h, frame_w = result_frame.shape[:2]
                        cy_end = min(frame_h, cy + ch)
                        cx_end = min(frame_w, cx + cw)
                        face_h = cy_end - cy
                        face_w = cx_end - cx

                        if face_h > 0 and face_w > 0 and cx >= 0 and cy >= 0:
                            # Resize face to fit available space if needed
                            if face_w != cw or face_h != ch:
                                face_resized = cv2.resize(face_final, (face_w, face_h), interpolation=cv2.INTER_CUBIC)
                            else:
                                face_resized = face_final

                            result_frame[cy:cy_end, cx:cx_end] = face_resized
                            print(f"Placed face at crop region ({cx}, {cy}) size ({face_w}, {face_h}) [orig face: ({x}, {y}, {w}, {h})]")
                        else:
                            print(f"Invalid crop placement bounds: cx={cx}, cy={cy}, cw={cw}, ch={ch}, face_w={face_w}, face_h={face_h}")
                    except Exception as e:
                        print(f"Failed to place face: {e}")
                else:
                    print(f"Invalid crop resize parameters: cw={cw}, ch={ch}, crop_face.size={crop_face.size if crop_face is not None else 'None'}")

                # Scale result back to original resolution if needed
                if self.resize_factor > 1:
                    result_frame = cv2.resize(result_frame, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)

                output_frames.append(result_frame)
        
        # Write output video
        print("Writing output video...")
        self.write_video_with_audio(output_frames, audio_path, output_path, fps)
        print(f"✓ Output saved: {output_path}")
        
        # Clear memory to prevent OOM
        self.cleanup_memory()
    
    def write_video_with_audio(self, frames, audio_path, output_path, fps):
        """Write video frames and merge with audio"""
        import uuid
        # ランダムな一時ファイル名を生成
        temp_video = output_path.replace('.mp4', f'_temp_{uuid.uuid4().hex[:8]}.mp4')
        
        # Keep original resolution (no forced upscaling)
        first_frame = frames[0]
        height, width = first_frame.shape[:2]
        print(f"Output resolution: {width}x{height} (original preserved)")
        
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
    
    def cleanup_memory(self):
        """Clear GPU and CPU memory after processing"""
        import gc
        
        # Clear Python garbage collection
        gc.collect()
        
        # Clear GPU memory if using CUDA
        if self.device == 'cuda':
            try:
                import torch
                if torch.cuda.is_available():
                    # Clear PyTorch CUDA cache
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Reset CUDA memory stats
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                    
                    print("✓ GPU memory cleared")
            except Exception as e:
                print(f"Warning: Could not clear GPU memory: {e}")
        
        # Clear ONNX Runtime session caches if possible
        try:
            # Force garbage collection on ONNX models
            if hasattr(self, 'wav2lip_model'):
                del self.wav2lip_model
            if hasattr(self, 'gfpgan_model'):
                del self.gfpgan_model
            
            # Reinitialize models for next use
            self.load_models()
            if self.initial_use_gfpgan:
                self.use_gfpgan = self.initial_use_gfpgan
                self.load_gfpgan()
                
            print("✓ ONNX models reloaded")
        except Exception as e:
            print(f"Warning: Could not reload models: {e}")
        
        # Final garbage collection
        gc.collect()
        print("✓ Memory cleanup completed")