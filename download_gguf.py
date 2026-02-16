from huggingface_hub import hf_hub_download
import os

def download_model():
    local_dir = "models"
    os.makedirs(local_dir, exist_ok=True)

    # 1. Download Main Model (Unsloth)
    repo_id_main = "unsloth/Qwen3-VL-8B-Instruct-GGUF"
    filename_main = "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
    
    print(f"Downloading {filename_main} from {repo_id_main}...")
    path_main = hf_hub_download(
        repo_id=repo_id_main,
        filename=filename_main,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Main model downloaded to: {path_main}")

    # 2. Download Vision Projector (mmproj) - REQUIRED for Vision
    repo_id_proj = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
    filename_proj = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"

    print(f"Downloading {filename_proj} from {repo_id_proj}...")
    path_proj = hf_hub_download(
        repo_id=repo_id_proj,
        filename=filename_proj,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Vision projector downloaded to: {path_proj}")

    return path_main

if __name__ == "__main__":
    download_model()
