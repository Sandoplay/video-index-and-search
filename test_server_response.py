from src.vision_analyzer import VisionAnalyzer
import requests
import json

def main():
    with VisionAnalyzer(model_id="models/Qwen3.5-9B-Q8_0.gguf") as analyzer:
        payload = {
            "messages": [
                {"role": "user", "content": "Hello, can you explain the history of the Eiffel Tower in 2 sentences?"}
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
        print("Sending request...")
        response = requests.post(analyzer.api_url, json=payload, timeout=120)
        print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    main()
