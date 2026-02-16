import os
import subprocess
import sys

def list_gguf_models():
    model_dir = "models"
    if not os.path.exists(model_dir):
        return []
    models = [f for f in os.listdir(model_dir) if f.endswith(".gguf") and "mmproj" not in f.lower()]
    
    # Sort: Qwen3 models first, then the rest alphabetically
    models.sort(key=lambda x: (not x.startswith("Qwen3"), x))
    return models

def find_mmproj(model_name):
    model_dir = "models"
    # Logic to match mmproj to model
    if "Qwen3" in model_name:
        proj = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
    elif "Qwen2.5" in model_name or "Qwen2" in model_name:
        proj = "mmproj-model-f16.gguf"
    else:
        proj = None
    
    if proj and os.path.exists(os.path.join(model_dir, proj)):
        return os.path.join(model_dir, proj)
    return None

def start_server():
    models = list_gguf_models()
    if not models:
        print("Error: No GGUF models found in 'models/' directory.")
        return

    print("\n=== Available Models (Qwen3 prioritized) ===")
    for i, m in enumerate(models):
        indicator = " (DEFAULT)" if i == 0 else ""
        print(f"[{i+1}] {m}{indicator}")
    
    try:
        choice_str = input(f"\nSelect model number [1-{len(models)}] (default 1): ").strip()
        choice = int(choice_str) - 1 if choice_str else 0
        if choice < 0 or choice >= len(models):
            print("Invalid selection, using default Qwen3.")
            choice = 0
    except ValueError:
        choice = 0

    selected_model = models[choice]
    model_path = os.path.abspath(os.path.join("models", selected_model))
    mmproj_path = find_mmproj(selected_model)
    
    server_exe = os.path.abspath(os.path.join("bin", "llama-cpp", "llama-server.exe"))
    
    cmd = [
        server_exe,
        "--model", model_path,
        "--n-gpu-layers", "-1",
        "--ctx-size", "50536",
        "--port", "8080",
        "--parallel", "1"
    ]

    if mmproj_path:
        print(f">>> Vision model detected. Using projector: {os.path.basename(mmproj_path)}")
        cmd.extend(["--mmproj", os.path.abspath(mmproj_path)])
    
    env = os.environ.copy()
    torch_lib = os.path.abspath(os.path.join(".venv", "Lib", "site-packages", "torch", "lib"))
    env["PATH"] = torch_lib + os.pathsep + env.get("PATH", "")

    print(f"\n>>> Starting server for: {selected_model}")
    print(f">>> Web UI: http://127.0.0.1:8080")
    print(">>> Press Ctrl+C to stop.\n")

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n>>> Server stopped.")

if __name__ == "__main__":
    start_server()
