import os
import base64
import time
import subprocess
import requests
import json
import signal
import sys
from typing import List
from .models import FrameMetadata

class VisionAnalyzer:
    def __init__(self, model_id: str = "models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", load_in_4bit: bool = False):
        self.model_path = model_id
        # Define mmproj path (expected to be in the same directory or models/)
        self.mmproj_path = "models/mmproj-model-f16.gguf"
        self.server_process = None
        self.api_url = "http://127.0.0.1:8080/v1/chat/completions"
        self.torch_lib = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages", "torch", "lib")
        self.is_external_server = False

    def _is_server_ready(self):
        """Checks if llama-server is ready and has loaded the model."""
        try:
            response = requests.get("http://127.0.0.1:8080/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "ok"
            return False
        except:
            return False

    def _start_server(self):
        """Starts the llama-server.exe with vision support."""
        if self._is_server_ready():
            print(">>> Connected to an existing and READY llama-server.")
            self.is_external_server = True
            return

        actual_model_path = os.path.abspath(self.model_path)
        actual_mmproj_path = os.path.abspath(self.mmproj_path)
        
        if not os.path.exists(actual_mmproj_path):
             actual_mmproj_path = os.path.abspath(os.path.join("models", os.path.basename(self.mmproj_path)))
        
        server_exe = os.path.abspath(os.path.join("bin", "llama-cpp", "llama-server.exe"))
        
        print(f">>> Starting llama-server with VISION support...")
        
        env = os.environ.copy()
        env["PATH"] = self.torch_lib + os.pathsep + env.get("PATH", "")
        
        cmd = [
            server_exe,
            "--model", actual_model_path,
            "--mmproj", actual_mmproj_path, # REQUIRED for vision!
            "--n-gpu-layers", "-1",
            "--port", "8080",
            "--ctx-size", "4096",
            "--n-predict", "512"
        ]
        
        self.server_process = subprocess.Popen(cmd, env=env)
        
        print(">>> Waiting for model and vision encoder to load...")
        max_retries = 60
        for i in range(max_retries):
            if self._is_server_ready():
                print(">>> Server ready with Vision support!")
                time.sleep(2)
                return
            if self.server_process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.server_process.returncode}")
            time.sleep(1)
        
        raise RuntimeError("Failed to start llama-server with vision support.")

    def _stop_server(self):
        if self.server_process and not self.is_external_server:
            print(">>> Stopping llama-server...")
            if os.name == 'nt':
                os.kill(self.server_process.pid, signal.CTRL_BREAK_EVENT)
            else:
                self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except:
                self.server_process.kill()
            self.server_process = None

    def _image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def analyze_frames(self, frames: List[FrameMetadata], transcription: str, batch_size: int = 1) -> List[FrameMetadata]:
        try:
            self._start_server()
            
            max_context_chars = 1000
            context = transcription[:max_context_chars] + "..." if len(transcription) > max_context_chars else transcription

            for i, frame in enumerate(frames):
                print(f"Processing frame {i+1}/{len(frames)} at {frame.timestamp:.2f}s...")
                
                image_b64 = self._image_to_base64(frame.path)
                
                # Use standard OpenAI-like format for Vision
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Audio context: {context}\n\nTask: Describe this visual scene in one detailed sentence."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                            ]
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 120
                }

                start_time = time.time()
                try:
                    response = requests.post(self.api_url, json=payload, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                    description = data["choices"][0]["message"]["content"].strip()
                    
                    elapsed = time.time() - start_time
                    tokens = data.get("usage", {}).get("completion_tokens", 0)
                    speed = tokens / elapsed if elapsed > 0 else 0
                    
                    print(f"  -> {description[:100]}... ({speed:.1f} t/s)")
                    frame.description = description
                except Exception as e:
                    print(f"  -> Error: {e}")
                    continue
                
            return frames
        finally:
            self._stop_server()
