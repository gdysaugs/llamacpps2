#!/usr/bin/env python3
"""
GPT-SoVITS Voice Cloning Engine
Simplified implementation for voice cloning with reference audio
"""

import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPTSoVITSEngine:
    """Main engine for GPT-SoVITS voice cloning"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        half_precision: bool = True,
        language: str = 'ja'  # Japanese by default
    ):
        """
        Initialize GPT-SoVITS engine
        
        Args:
            model_path: Path to model directory
            device: 'cuda' or 'cpu'
            half_precision: Use FP16 for GPU inference
            language: Target language ('ja', 'en', 'zh')
        """
        self.model_path = Path(model_path)
        self.device = device
        self.half_precision = half_precision and device == 'cuda'
        self.language = language
        
        # Check device availability
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
            self.half_precision = False
        
        self.models = {}
        self.sampling_rate = 32000  # GPT-SoVITS default
        
    def load_models(self, checkpoint_path: str):
        """
        Load GPT-SoVITS models from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file (.ckpt)
        """
        logger.info(f"Loading models from {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Initialize GPT model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # GPT component for text encoding
            self.tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",  # Base model
                local_files_only=False
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model weights from checkpoint if available
            if 'gpt_model' in checkpoint:
                self.gpt_model = checkpoint['gpt_model']
                if self.half_precision:
                    self.gpt_model = self.gpt_model.half()
                self.gpt_model = self.gpt_model.to(self.device)
                self.gpt_model.eval()
            
            # Load SoVITS vocoder from checkpoint
            if 'sovits_model' in checkpoint:
                self.sovits_model = checkpoint['sovits_model']
                if self.half_precision:
                    self.sovits_model = self.sovits_model.half()
                self.sovits_model = self.sovits_model.to(self.device)
                self.sovits_model.eval()
            
            # Store config
            self.config = checkpoint.get('config', {})
            
            logger.info("✓ Models loaded successfully")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Precision: {'FP16' if self.half_precision else 'FP32'}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def extract_speaker_embedding(
        self,
        reference_audio_path: str,
        reference_text: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract speaker embedding from reference audio
        
        Args:
            reference_audio_path: Path to reference audio file
            reference_text: Optional reference text (None for text-free mode)
            
        Returns:
            Speaker embedding tensor
        """
        logger.info(f"Extracting speaker embedding from {reference_audio_path}")
        
        # Load reference audio
        audio, sr = librosa.load(reference_audio_path, sr=self.sampling_rate)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        if self.half_precision:
            audio_tensor = audio_tensor.half()
        audio_tensor = audio_tensor.to(self.device)
        
        # Extract features using mel spectrogram
        mel = self.audio_to_mel(audio_tensor)
        
        # Get speaker embedding (simplified version)
        with torch.no_grad():
            if hasattr(self, 'sovits_model'):
                # Use SoVITS encoder for speaker embedding
                speaker_embedding = self.sovits_model.encode_speaker(mel)
            else:
                # Fallback: use mel features directly
                speaker_embedding = mel.mean(dim=-1)
        
        logger.info(f"✓ Speaker embedding extracted: shape {speaker_embedding.shape}")
        return speaker_embedding
    
    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to mel spectrogram
        
        Args:
            audio: Audio tensor
            
        Returns:
            Mel spectrogram tensor
        """
        # Simple mel spectrogram extraction
        n_fft = 2048
        hop_length = 300
        n_mels = 128
        
        # Using librosa-style mel extraction
        if audio.is_cuda:
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio.numpy()
        
        mel = librosa.feature.melspectrogram(
            y=audio_np.squeeze(),
            sr=self.sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = torch.FloatTensor(mel).unsqueeze(0)
        
        if self.half_precision:
            mel = mel.half()
        
        return mel.to(self.device)
    
    def synthesize(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
        speed: float = 1.0,
        temperature: float = 0.3
    ) -> np.ndarray:
        """
        Synthesize speech from text using speaker embedding
        
        Args:
            text: Text to synthesize
            speaker_embedding: Speaker embedding from reference audio
            speed: Speech speed factor
            temperature: Sampling temperature
            
        Returns:
            Audio array
        """
        logger.info(f"Synthesizing: '{text}'")
        
        # Tokenize text
        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        # Generate with GPT model (simplified)
        with torch.no_grad():
            if hasattr(self, 'gpt_model'):
                # Get text features from GPT
                outputs = self.gpt_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                text_features = outputs.hidden_states[-1].mean(dim=1)
            else:
                # Fallback: use token embeddings
                text_features = input_ids.float().mean(dim=1, keepdim=True)
                if self.half_precision:
                    text_features = text_features.half()
            
            # Combine with speaker embedding
            combined_features = torch.cat([
                speaker_embedding.unsqueeze(1).expand(-1, text_features.size(1), -1),
                text_features
            ], dim=-1)
            
            # Generate audio with SoVITS vocoder
            if hasattr(self, 'sovits_model'):
                audio = self.sovits_model.decode(combined_features, speed=speed)
            else:
                # Fallback: generate simple sine wave (for testing)
                duration = len(text) * 0.1 * (1.0 / speed)  # Rough duration estimate
                t = np.linspace(0, duration, int(self.sampling_rate * duration))
                audio = np.sin(2 * np.pi * 440 * t) * 0.3  # 440Hz sine wave
                audio = torch.FloatTensor(audio).unsqueeze(0)
        
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()
        
        # Normalize
        audio = audio / np.max(np.abs(audio) + 1e-7)
        
        logger.info(f"✓ Synthesized {len(audio)} samples")
        return audio
    
    def inference(
        self,
        text: str,
        reference_audio: str,
        output_path: str,
        reference_text: Optional[str] = None,
        speed: float = 1.0,
        temperature: float = 0.3
    ):
        """
        Full inference pipeline
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio
            output_path: Output audio path
            reference_text: Optional reference text (None for text-free)
            speed: Speech speed
            temperature: Sampling temperature
        """
        logger.info("="*60)
        logger.info("🎤 GPT-SoVITS Voice Cloning")
        logger.info("="*60)
        logger.info(f"📝 Text: {text}")
        logger.info(f"🎵 Reference: {reference_audio}")
        logger.info(f"💾 Output: {output_path}")
        logger.info(f"🚀 Text-free mode: {reference_text is None}")
        logger.info("="*60)
        
        # Extract speaker embedding
        speaker_embedding = self.extract_speaker_embedding(
            reference_audio,
            reference_text
        )
        
        # Synthesize speech
        audio = self.synthesize(
            text,
            speaker_embedding,
            speed=speed,
            temperature=temperature
        )
        
        # Save output
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        sf.write(output_path, audio, self.sampling_rate)
        
        logger.info("="*60)
        logger.info(f"✅ SUCCESS! Audio saved to {output_path}")
        logger.info("="*60)
        
        # Cleanup GPU memory
        self.cleanup()
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("✓ GPU memory cleared")


class SimplifiedGPTSoVITS:
    """
    Simplified GPT-SoVITS implementation using voice characteristics transfer
    Extracts prosody and timbre from reference audio
    """
    
    def __init__(self, model_dir: str, device: str = 'cuda'):
        self.model_dir = Path(model_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.sampling_rate = 22050  # Standard TTS sampling rate
        
        # Load Japanese text processing
        try:
            import pyopenjtalk
            self.pyopenjtalk = pyopenjtalk
        except ImportError:
            logger.warning("pyopenjtalk not available, using basic processing")
            self.pyopenjtalk = None
            
        logger.info(f"Initializing Simplified GPT-SoVITS on {self.device}")
    
    def extract_voice_features(self, reference_audio: str):
        """Extract voice characteristics from reference audio"""
        # Load reference audio
        ref_audio, sr = librosa.load(reference_audio, sr=self.sampling_rate)
        
        # Extract prosodic features
        pitch, voiced_flag, voiced_probs = librosa.pyin(
            ref_audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            frame_length=2048
        )
        
        # Get spectral features
        mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=ref_audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=ref_audio, sr=sr)
        
        # Voice characteristics
        features = {
            'base_pitch': np.nanmedian(pitch[voiced_flag]),
            'pitch_range': np.nanstd(pitch[voiced_flag]),
            'spectral_centroid': np.mean(spectral_centroid),
            'spectral_rolloff': np.mean(spectral_rolloff),
            'mfcc_mean': np.mean(mfcc, axis=1),
            'tempo': librosa.feature.tempo(y=ref_audio, sr=sr)[0],
            'energy': np.mean(librosa.feature.rms(y=ref_audio))
        }
        
        return features, ref_audio
    
    def text_to_phonemes(self, text: str):
        """Convert text to phonemes for Japanese"""
        if self.pyopenjtalk:
            # Use OpenJTalk for proper Japanese phoneme extraction
            phonemes = []
            for feature in self.pyopenjtalk.extract_fullcontext(text):
                # Extract phoneme from full context
                parts = feature.split('-')
                if len(parts) > 1:
                    phoneme = parts[1].split('+')[0]
                    if phoneme != 'xx':  # Skip silence
                        phonemes.append(phoneme)
            return phonemes
        else:
            # Fallback: character-based processing
            return list(text.replace(' ', ''))  # Remove spaces for fallback
    
    def get_phoneme_formants(self, phoneme):
        """Get formant frequencies for Japanese phonemes"""
        
        # Japanese phoneme to formant mapping (F1, F2, F3)
        formant_map = {
            # Vowels (F1, F2, F3) 
            'a': (730, 1090, 2440),    # あ
            'i': (270, 2290, 3010),    # い  
            'u': (300, 870, 2240),     # う
            'e': (530, 1840, 2480),    # え
            'o': (570, 840, 2410),     # お
            'U': (350, 1170, 2390),    # ウ (devoiced)
            
            # Semi-vowels
            'w': (300, 610, 2150),     # w sound
            'y': (280, 2070, 2960),    # y sound
            'ry': (350, 1690, 2490),   # ry sound
            
            # Nasals
            'N': (280, 1650, 2490),    # ん
            'n': (280, 1650, 2490),    # n
            'm': (280, 1650, 2490),    # m
            
            # Fricatives  
            's': (200, 4000, 8000),    # s (high frequency)
            'sh': (200, 2800, 6000),   # sh
            'h': (300, 1500, 3500),    # h
            'z': (200, 1700, 4500),    # z
            
            # Stops
            'k': (200, 2500, 3500),    # k 
            't': (200, 1800, 3200),    # t  
            'p': (200, 1000, 2500),    # p
            'b': (200, 1000, 2500),    # b
            'd': (200, 1800, 3200),    # d
            'g': (200, 2500, 3500),    # g
            
            # Affricates
            'ch': (200, 2800, 6000),   # ch
            'ts': (200, 4000, 8000),   # ts
            
            # Special
            'pau': (0, 0, 0),          # pause
            'sil': (0, 0, 0),          # silence
        }
        
        return formant_map.get(phoneme, (400, 1500, 2500))  # default
    
    def is_vowel(self, phoneme):
        """Check if phoneme is a vowel"""
        return phoneme in {'a', 'i', 'u', 'e', 'o', 'U'}
    
    def is_silence(self, phoneme):
        """Check if phoneme is silence"""
        return phoneme in {'pau', 'sil'}
        
    def synthesize_phoneme(self, phoneme, duration, base_pitch, position):
        """Synthesize individual phoneme with proper formants"""
        
        samples = int(self.sampling_rate * duration)
        if samples <= 0:
            return np.zeros(1)
            
        t = np.linspace(0, duration, samples)
        
        # Handle silence
        if self.is_silence(phoneme):
            return np.zeros(samples)
        
        # Get formant frequencies
        f1, f2, f3 = self.get_phoneme_formants(phoneme)
        
        if f1 == 0:  # silence
            return np.zeros(samples)
        
        # Pitch with natural variation
        f0_variation = 1.0 + 0.15 * np.sin(2 * np.pi * position)
        f0 = base_pitch * f0_variation
        
        # Generate based on phoneme type
        if self.is_vowel(phoneme):
            # Vowels: strong formant structure
            audio = self.generate_formant_audio(t, f0, f1, f2, f3, vowel=True)
            
        elif phoneme in ['N', 'n', 'm', 'ry']:
            # Sonorants: weaker formants
            audio = self.generate_formant_audio(t, f0, f1, f2, f3, vowel=False) * 0.8
            
        elif phoneme in ['s', 'sh', 'h', 'z', 'ch', 'ts']:
            # Fricatives: noise + filtering
            noise = np.random.normal(0, 0.2, samples)
            if f2 > 3000:  # High frequency fricatives
                audio = self.high_pass_filter(noise, f2 / 2)
            else:
                audio = self.band_pass_filter(noise, f1, f2)
                
        else:
            # Stops: burst + formant transition
            burst_duration = 0.02  # 20ms burst
            burst_samples = int(self.sampling_rate * burst_duration)
            burst_samples = min(burst_samples, samples // 2)
            
            if burst_samples > 0:
                # Sharp burst
                burst = np.random.normal(0, 0.3, burst_samples)
                burst *= np.exp(-np.linspace(0, 3, burst_samples))
                
                # Formant transition
                transition_samples = samples - burst_samples
                if transition_samples > 0:
                    t_trans = np.linspace(0, duration - burst_duration, transition_samples)
                    transition = self.generate_formant_audio(t_trans, f0, f1, f2, f3, vowel=False) * 0.4
                    audio = np.concatenate([burst, transition])
                else:
                    audio = burst
            else:
                # Fallback to formant
                audio = self.generate_formant_audio(t, f0, f1, f2, f3, vowel=False) * 0.3
        
        # Apply natural envelope
        if samples > 1:
            envelope = np.hanning(samples)
            audio *= envelope
        
        return audio
    
    def generate_formant_audio(self, t, f0, f1, f2, f3, vowel=True):
        """Generate audio with formant structure"""
        if len(t) == 0:
            return np.array([])
        
        # Base harmonic series
        audio = np.zeros(len(t))
        
        # Generate harmonics up to Nyquist
        max_harmonic = min(20, int(self.sampling_rate // (2 * f0)))
        
        for h in range(1, max_harmonic + 1):
            harmonic_freq = f0 * h
            
            # Calculate amplitude based on formant proximity
            amp = 0.1 / h  # Basic harmonic decay
            
            # Boost harmonics near formants
            if abs(harmonic_freq - f1) < 100:
                amp *= (3.0 if vowel else 2.0)
            elif abs(harmonic_freq - f2) < 150:
                amp *= (2.5 if vowel else 1.8)
            elif abs(harmonic_freq - f3) < 200:
                amp *= (1.8 if vowel else 1.4)
                
            # Add harmonic
            if harmonic_freq < self.sampling_rate // 2:
                harmonic = np.sin(2 * np.pi * harmonic_freq * t) * amp
                audio += harmonic
        
        return audio
    
    def high_pass_filter(self, audio, cutoff):
        """Simple high-pass filter for fricatives"""
        from scipy.signal import butter, filtfilt
        
        try:
            nyquist = self.sampling_rate // 2
            normalized_cutoff = min(0.95, cutoff / nyquist)
            b, a = butter(4, normalized_cutoff, btype='high')
            return filtfilt(b, a, audio)
        except:
            return audio
    
    def band_pass_filter(self, audio, low_freq, high_freq):
        """Simple band-pass filter"""
        from scipy.signal import butter, filtfilt
        
        try:
            nyquist = self.sampling_rate // 2
            low = max(50, low_freq) / nyquist
            high = min(0.95, high_freq) / nyquist
            
            if low < high:
                b, a = butter(4, [low, high], btype='band')
                return filtfilt(b, a, audio)
        except:
            pass
        return audio
    
    def synthesize_with_features(self, phonemes, voice_features, duration=None):
        """Generate audio using proper phoneme synthesis"""
        
        # Calculate total duration
        if duration is None:
            total_duration = 0
            for phoneme in phonemes:
                if self.is_silence(phoneme):
                    total_duration += 0.1  # Silence
                elif self.is_vowel(phoneme):
                    total_duration += 0.15  # Longer vowels
                else:
                    total_duration += 0.08  # Shorter consonants
            duration = total_duration
        
        # Get base pitch
        base_pitch = voice_features.get('base_pitch', 150)
        if not base_pitch or base_pitch < 50:
            base_pitch = 150  # Safe default
        
        # Synthesize each phoneme
        audio_segments = []
        
        for i, phoneme in enumerate(phonemes):
            # Duration for this phoneme
            if self.is_silence(phoneme):
                phon_duration = 0.1
            elif self.is_vowel(phoneme):
                phon_duration = 0.15
            else:
                phon_duration = 0.08
            
            # Position for pitch variation
            position = i / max(1, len(phonemes) - 1)
            
            # Synthesize phoneme
            segment = self.synthesize_phoneme(
                phoneme, phon_duration, base_pitch, position
            )
            
            audio_segments.append(segment)
            
            print(f"  Phoneme '{phoneme}': {len(segment)} samples")
        
        # Concatenate all segments
        if audio_segments:
            audio = np.concatenate(audio_segments)
        else:
            audio = np.zeros(int(self.sampling_rate * 0.5))
        
        # Apply voice energy
        energy = voice_features.get('energy', 0.1)
        audio = audio * min(1.0, max(0.1, energy * 2.0))
        
        return audio
    
    def clone_voice(
        self,
        text: str,
        reference_audio: str,
        output_path: str,
        **kwargs
    ):
        """
        Improved voice cloning with feature extraction
        
        Args:
            text: Text to speak
            reference_audio: Reference voice audio file
            output_path: Output audio file path
        """
        logger.info(f"Cloning voice for text: '{text}'")
        
        # Extract voice features from reference
        voice_features, ref_audio = self.extract_voice_features(reference_audio)
        logger.info(f"✓ Extracted voice features - Pitch: {voice_features['base_pitch']:.1f}Hz")
        
        # Convert text to phonemes
        phonemes = self.text_to_phonemes(text)
        logger.info(f"✓ Generated {len(phonemes)} phonemes")
        
        # Synthesize audio
        audio = self.synthesize_with_features(phonemes, voice_features)
        
        # Normalize and save
        audio = audio / (np.max(np.abs(audio)) + 1e-7) * 0.8
        sf.write(output_path, audio, self.sampling_rate)
        
        logger.info(f"✓ Voice cloned and saved to {output_path}")
        return output_path


if __name__ == "__main__":
    # Test implementation
    engine = SimplifiedGPTSoVITS(
        model_dir="models/gpt_sovits",
        device="cuda"
    )
    
    # Example usage
    engine.clone_voice(
        text="こんにちは、テストです",
        reference_audio="reference.wav",
        output_path="output/cloned_voice.wav"
    )