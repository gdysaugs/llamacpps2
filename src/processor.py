import os
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import tempfile
import subprocess
from pathlib import Path

class Wav2LipProcessor:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        
        # Load models
        self.load_models(model_path)
        
    def load_models(self, model_path):
        """Load ONNX models"""
        print("Loading models...")
        
        # Face detection model
        recognition_path = os.path.join(model_path, 'wav2lip', 'recognition.onnx')
        if os.path.exists(recognition_path):
            self.face_detector = ort.InferenceSession(recognition_path, providers=self.providers)
            print(f"✓ Face detection model loaded")
        
        # Segmentation model
        xseg_path = os.path.join(model_path, 'wav2lip', 'xseg.onnx')
        if os.path.exists(xseg_path):
            self.segmentation = ort.InferenceSession(xseg_path, providers=self.providers)
            print(f"✓ Segmentation model loaded")
            
        # Blending model
        blend_path = os.path.join(model_path, 'wav2lip', 'blendmasker.onnx')
        if os.path.exists(blend_path):
            self.blender = ort.InferenceSession(blend_path, providers=self.providers)
            print(f"✓ Blending model loaded")
            
        # Denoiser model
        denoiser_path = os.path.join(model_path, 'wav2lip', 'denoiser_fp16.onnx')
        if os.path.exists(denoiser_path):
            self.denoiser = ort.InferenceSession(denoiser_path, providers=self.providers)
            print(f"✓ Denoiser model loaded")
            
    def extract_audio(self, video_path, output_path):
        """Extract audio from video"""
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            output_path, '-y'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
        
    def process_video(self, video_path, audio_path, output_path, use_gfpgan=False):
        """Main processing function"""
        print(f"Processing video: {video_path}")
        print(f"Using audio: {audio_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create temp directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_video = os.path.join(temp_dir, 'temp_video.mp4')
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
            
            # Process frames
            print(f"Processing {total_frames} frames...")
            with tqdm(total=total_frames) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Here we would apply lip-sync processing
                    # For now, just pass through the frame
                    processed_frame = self.process_frame(frame)
                    
                    out.write(processed_frame)
                    pbar.update(1)
            
            cap.release()
            out.release()
            
            # Merge audio and video
            print("Merging audio and video...")
            self.merge_audio_video(temp_video, audio_path, output_path)
            
        print(f"✓ Output saved to: {output_path}")
        
    def process_frame(self, frame):
        """Process single frame (placeholder for now)"""
        # TODO: Implement actual lip-sync processing
        return frame
        
    def merge_audio_video(self, video_path, audio_path, output_path):
        """Merge audio and video using ffmpeg"""
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-c:v', 'libx264', '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0',
            output_path, '-y'
        ]
        subprocess.run(cmd, check=True, capture_output=True)

class GFPGANEnhancer:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        
        # Load GFPGAN model
        gfpgan_path = os.path.join(model_path, 'gfpgan', 'GFPGANv1.4.onnx')
        if os.path.exists(gfpgan_path):
            self.model = ort.InferenceSession(gfpgan_path, providers=self.providers)
            print("✓ GFPGAN model loaded")
        else:
            print(f"Warning: GFPGAN model not found at {gfpgan_path}")
            self.model = None
            
    def enhance_face(self, face_img):
        """Enhance face using GFPGAN"""
        if self.model is None:
            return face_img
            
        # Preprocess
        img = cv2.resize(face_img, (512, 512))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # Run inference
        output = self.model.run(None, {'input': img})[0]
        
        # Postprocess
        output = np.squeeze(output)
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255).clip(0, 255).astype(np.uint8)
        
        # Resize back to original size
        output = cv2.resize(output, (face_img.shape[1], face_img.shape[0]))
        
        return output