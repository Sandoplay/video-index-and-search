import sys
sys.stdout.reconfigure(encoding='utf-8')
import whisper
import os
import subprocess
import torch
import gc

def extract_audio(video_path, audio_path):
    if os.path.exists(audio_path):
        return
    print(f">>> Extracting audio to {audio_path}...")
    ffmpeg_bin = os.path.join("bin", "ffmpeg.exe")
    if not os.path.exists(ffmpeg_bin): ffmpeg_bin = "ffmpeg"
    
    cmd = [ffmpeg_bin, "-y", "-i", video_path, "-ac", "1", "-ar", "16000", audio_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def test_config(model, audio_path, name, **kwargs):
    print(f"\n--- Testing Config: {name} ---")
    print(f"Params: {kwargs}")
    try:
        abs_path = os.path.abspath(audio_path)
        result = model.transcribe(abs_path, **kwargs)
        text = result["text"].strip()
        print(f"RESULT: {text}")
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

    results = {}

    # --- Test large-v3 ---
    print("\nLoading Whisper model: large-v3 into models/ directory...")
    os.makedirs("models", exist_ok=True)
    model_v3 = whisper.load_model("large-v3", download_root="models")

    configs_v3 = [
        {
            "name": "[v3] condition_on_previous_text=False (Current Best)",
            "params": {
                "language": "uk",
                "condition_on_previous_text": False
            }
        },
        {
            "name": "[v3] Stricter no_speech_threshold",
            "params": {
                "language": "uk",
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.3, # Default is 0.6, lower means it's more likely to skip noise
                "logprob_threshold": -0.8
            }
        },
        {
            "name": "[v3] without_timestamps=True",
            "params": {
                "language": "uk",
                "condition_on_previous_text": False,
                "without_timestamps": True
            }
        },
        {
            "name": "[v3] Word Timestamps",
            "params": {
                "language": "uk",
                "condition_on_previous_text": False,
                "word_timestamps": True
            }
        },
        {
            "name": "[v3] Higher compression_ratio_threshold",
            "params": {
                "language": "uk",
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 3.0 # Default is 2.4, maybe allows more text
            }
        }
    ]

    for cfg in configs_v3:
        res = test_config(model_v3, audio_path, cfg["name"], **cfg["params"])
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
