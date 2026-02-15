from huggingface_hub import hf_hub_download
import os

def download_mmproj():
    repo_id = "lmstudio-community/Qwen2.5-VL-3B-Instruct-GGUF"
    filename = "mmproj-model-f16.gguf"
    local_dir = "models"
    
    print(f"Downloading {filename} from {repo_id}...")
    os.makedirs(local_dir, exist_ok=True)
    
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Vision encoder downloaded to: {path}")
    return path

if __name__ == "__main__":
    download_mmproj()
