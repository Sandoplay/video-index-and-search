import whisper
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
import os
import subprocess
import torch
from pathlib import Path

def extract_audio(video_path, audio_path):
    if os.path.exists(audio_path):
        return
    print(f">>> Extracting audio to {audio_path}...")
    ffmpeg_bin = os.path.join("bin", "ffmpeg.exe")
    if not os.path.exists(ffmpeg_bin): ffmpeg_bin = "ffmpeg"
    
    cmd = [ffmpeg_bin, "-y", "-i", video_path, "-ac", "1", "-ar", "16000", audio_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def detect_language(model, audio_path):
    abs_path = os.path.abspath(audio_path)
    audio = load_audio(abs_path)
    
    # Sample up to 3 segments (start, middle, end) of 30s each
    segment_length = 30 * whisper.audio.SAMPLE_RATE
    total_samples = len(audio)
    
    sample_points = [0]
    if total_samples > segment_length * 2:
        sample_points.append(total_samples // 2 - segment_length // 2)
        sample_points.append(total_samples - segment_length)
    elif total_samples > segment_length:
        sample_points.append(total_samples - segment_length)
    
    all_probs = []
    n_mels = model.dims.n_mels
    
    for start in sample_points:
        end = min(start + segment_length, total_samples)
        segment = audio[start:end]
        segment = pad_or_trim(segment)
        mel = log_mel_spectrogram(segment, n_mels=n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        all_probs.append(probs)
    
    # Average probabilities across segments
    avg_probs = {}
    for lang in all_probs[0].keys():
        avg_probs[lang] = sum(p[lang] for p in all_probs) / len(all_probs)
    
    lang = max(avg_probs, key=avg_probs.get)
    confidence = avg_probs[lang]
    print(f"   [Auto-Detect] Sampled {len(all_probs)} segments. Detected: '{lang}' (Avg Confidence: {confidence:.2f})")
    
    # Log top 3 candidates if confidence is below 90%
    if confidence < 0.9:
        sorted_probs = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
        top_3 = ", ".join([f"{l}: {p:.2f}" for l, p in sorted_probs[:3]])
        print(f"   [Auto-Detect] Top candidates: {top_3}")
        
    return lang

def test_config(model, audio_path, name, **kwargs):
    print(f"\n--- Testing Config: {name} ---")
    
    # Handle dynamic language detection simulation
    if kwargs.get("language") == "AUTO_DETECT":
        detected_lang = detect_language(model, audio_path)
        kwargs["language"] = detected_lang
        print(f"   -> Locking language to: {detected_lang}")

    print(f"Params: {kwargs}")
    try:
        abs_path = os.path.abspath(audio_path)
        result = model.transcribe(abs_path, **kwargs)
        text = result["text"].strip()
        print(f"RESULT (first 200 chars): {text[:200]}...")
        return text
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    video_path = "data/input/_stalker__meme_-20250309-0001.mp4"
    audio_path = "data/temp_test_audio.wav"
    
    if not os.path.exists(video_path):
        print(f"Error: Video {video_path} not found!")
        return

    os.makedirs("data", exist_ok=True)
    extract_audio(video_path, audio_path)

    print("Loading Whisper model: large-v3...")
    model = whisper.load_model("large-v3")

    configs = [
        {
            "name": "Auto-Detect Mode",
            "params": {
                "language": "AUTO_DETECT",
                "best_of": 10,
                "fp16": torch.cuda.is_available()
            }
        },
        {
            "name": "Hardcoded Ukrainian",
            "params": {
                "language": "uk"
            }
        }
    ]

    results = {}
    for cfg in configs:
        res = test_config(model, audio_path, cfg["name"], **cfg["params"])
        results[cfg["name"]] = res

    reference_text = "Ось чому у наших снах бачимо їхні спогади. Заради цього я втратив голову разом з усім іншим. Хаха. Варта тук-тук-тук. Ой а що це за звуки такі гучні довкола, га? Чуєте? Увага, здається це зона цвяхи забиває у ваші гроби. Хохохохо. Не подобається? Це лише початок."

    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"[Reference (Manual)]:\n{reference_text}\n")
    for name, text in results.items():
        print(f"[{name}]:\n{text}\n")

if __name__ == "__main__":
    main()
