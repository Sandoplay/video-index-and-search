from huggingface_hub import HfApi

def list_wheels():
    api = HfApi()
    files = api.list_repo_files("dougeeai/llama-cpp-python-wheels")
    print("Found wheels:")
    for f in files:
        if "cp313" in f and "win_amd64" in f:
            print(f)

if __name__ == "__main__":
    list_wheels()
