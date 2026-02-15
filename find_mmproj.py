from huggingface_hub import HfApi

def list_files():
    api = HfApi()
    files = api.list_repo_files("lmstudio-community/Qwen2.5-VL-3B-Instruct-GGUF")
    print("Files in repo:")
    for f in files:
        print(f)

if __name__ == "__main__":
    list_files()
