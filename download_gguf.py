from huggingface_hub import hf_hub_download
import os

def download_model():
    # Using lmstudio-community version as it's verified to work
    repo_id = "lmstudio-community/Qwen2.5-VL-3B-Instruct-GGUF"
    filename = "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    local_dir = "models"
    
    print(f"Downloading {filename} from {repo_id}...")
    os.makedirs(local_dir, exist_ok=True)
    
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to: {path}")
    return path

if __name__ == "__main__":
    download_model()
