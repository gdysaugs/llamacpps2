#!/usr/bin/env python3
"""
Debug script for phoneme processing
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

def test_phoneme_extraction():
    """Test phoneme extraction with pyopenjtalk"""
    
    # Test text
    test_text = "こんにちは、改良版テストです"
    
    print("="*50)
    print("📝 Phoneme Extraction Debug")
    print("="*50)
    print(f"Input text: {test_text}")
    print()
    
    # Test pyopenjtalk
    try:
        import pyopenjtalk
        
        print("🔍 PyOpenJTalk Full Context Analysis:")
        features = pyopenjtalk.extract_fullcontext(test_text)
        
        for i, feature in enumerate(features[:20]):  # Show first 20
            print(f"  {i:2d}: {feature}")
        print(f"  ... (total {len(features)} features)")
        print()
        
        print("🔍 PyOpenJTalk G2P (Grapheme to Phoneme):")
        try:
            phonemes = pyopenjtalk.g2p(test_text)
            print(f"  Phonemes: {phonemes}")
        except:
            phonemes = None
            print("  G2P not available")
        print()
        
        print("🔍 Manual Phoneme Extraction:")
        manual_phonemes = []
        for feature in features:
            # Extract phoneme from full context
            parts = feature.split('-')
            if len(parts) > 1:
                phoneme = parts[1].split('+')[0]
                if phoneme != 'xx' and phoneme != 'sil':  # Skip silence
                    manual_phonemes.append(phoneme)
                    
        print(f"  Extracted phonemes: {manual_phonemes}")
        print(f"  Count: {len(manual_phonemes)}")
        print()
        
        return manual_phonemes
        
    except ImportError as e:
        print(f"❌ pyopenjtalk not available: {e}")
        return None

def test_phoneme_to_frequency():
    """Test phoneme to frequency mapping"""
    
    # Common Japanese phoneme to formant mapping
    japanese_formants = {
        # Vowels
        'a': (730, 1090),   # あ
        'i': (270, 2290),   # い  
        'u': (300, 870),    # う
        'e': (530, 1840),   # え
        'o': (570, 840),    # お
        
        # Consonants (approximate)
        'k': (200, 2500),   # k sound
        'n': (200, 1500),   # n sound
        't': (200, 2000),   # t sound
        's': (200, 4000),   # s sound
        'h': (200, 3000),   # h sound
        'r': (200, 1200),   # r sound
        'm': (200, 1000),   # m sound
        
        # Default for unknown
        'xx': (300, 1500)
    }
    
    print("🎵 Phoneme to Formant Mapping:")
    for phoneme, (f1, f2) in japanese_formants.items():
        print(f"  {phoneme}: F1={f1}Hz, F2={f2}Hz")
    print()
    
    return japanese_formants

def debug_synthesis_process(phonemes):
    """Debug the synthesis process step by step"""
    
    if not phonemes:
        return
    
    print("🔧 Synthesis Process Debug:")
    print(f"  Input phonemes: {phonemes}")
    
    # Duration calculation
    duration = len(phonemes) * 0.15
    print(f"  Calculated duration: {duration:.2f}s")
    
    # Frequency calculation per phoneme
    base_freq = 150  # Default base frequency
    
    print("  Frequency per phoneme:")
    for i, phoneme in enumerate(phonemes):
        freq_mod = 1.0 + 0.3 * np.sin(2 * np.pi * i / len(phonemes))
        freq = base_freq * freq_mod
        print(f"    {i:2d}: '{phoneme}' -> {freq:.1f}Hz")
    
    print()

if __name__ == "__main__":
    import numpy as np
    
    # Run debug tests
    phonemes = test_phoneme_extraction()
    test_phoneme_to_frequency()
    debug_synthesis_process(phonemes)
    
    print("="*50)
    print("🚨 DIAGNOSIS:")
    print("="*50)
    
    if phonemes:
        unique_phonemes = set(phonemes)
        print(f"✓ {len(phonemes)} phonemes extracted")
        print(f"✓ {len(unique_phonemes)} unique phonemes: {unique_phonemes}")
        
        if len(unique_phonemes) == 1:
            print(f"❌ PROBLEM: All phonemes are the same: '{list(unique_phonemes)[0]}'")
            print("   This explains why you only hear one sound!")
        elif len(unique_phonemes) < 3:
            print(f"⚠️  WARNING: Very few unique phonemes ({len(unique_phonemes)})")
        else:
            print("✓ Phoneme extraction looks good")
            print("❌ Problem might be in synthesis logic")
    else:
        print("❌ PROBLEM: No phonemes extracted")
        print("   pyopenjtalk might not be working properly")