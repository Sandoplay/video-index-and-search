import os
import sys
from pathlib import Path

# Add PyTorch lib directory to DLL search path for Windows
# This allows llama-cpp-python to find cublas/cudart DLLs provided by PyTorch
current_dir = Path(__file__).parent.absolute()
venv_site_packages = current_dir / ".venv" / "Lib" / "site-packages"
torch_lib_path = venv_site_packages / "torch" / "lib"

if torch_lib_path.exists():
    print(f"Adding Torch lib to PATH: {torch_lib_path}")
    os.environ["PATH"] += os.pathsep + str(torch_lib_path)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(str(torch_lib_path))
else:
    print(f"Warning: Torch lib path not found at {torch_lib_path}")

from llama_cpp import Llama

def check_cuda():
    print("Checking llama-cpp-python CUDA support...")
    
    model_path = "models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        print("Model not found, cannot test load.")
        return

    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            verbose=True
        )
        print("\nModel loaded. Check the logs above for 'BLAS = 1' (good) or 'BLAS = 0' (bad).")
        print("Specifically look for 'ggml_cuda_init: found X CUDA devices'.")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_cuda()
