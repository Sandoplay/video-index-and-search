import os
import base64
import time
import subprocess
import requests
import json
import signal
import sys
from typing import List, Optional, Dict, Any
from .models import FrameMetadata

class VisionAnalyzer:
    # Constants based on Qwen3.5-VL and user testing on RTX 5060 Ti
    CONTEXT_SIZE = 72000 # Optimized for 16GB VRAM with Q8_0 quantization
    TOKENS_PER_IMAGE = 1024  # Adjusted for Qwen3.5-VL typical usage
    TOKENS_PER_TEXT_CHAR = 0.3 # Rough estimate
    RESERVE_TOKENS = 2000    # Increased for reasoning/thinking margin

    def __init__(self, model_id: str = "models/Qwen3.5-9B-Q8_0.gguf", load_in_4bit: bool = False, verbose: bool = False):
        self.model_path = model_id
        self.mmproj_path = "models/mmproj-F16.gguf"
        self.server_process = None
        self.api_url = "http://127.0.0.1:8080/v1/chat/completions"
        self.torch_lib = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages", "torch", "lib")
        self.is_external_server = False
        self.verbose = verbose

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_server()

    def _is_server_ready(self):
        try:
            response = requests.get("http://127.0.0.1:8080/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "ok"
            return False
        except:
            return False

    def _start_server(self):
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
        print(f"    Context size set to {self.CONTEXT_SIZE} for max throughput.")
        
        env = os.environ.copy()
        env["PATH"] = self.torch_lib + os.pathsep + env.get("PATH", "")
        
        cmd = [
            server_exe,
            "--model", actual_model_path,
            "--mmproj", actual_mmproj_path,
            "--n-gpu-layers", "-1",
            "--port", "8080",
            "--ctx-size", str(self.CONTEXT_SIZE),
            "--n-predict", "10240",
            "--parallel", "1",
            "--reasoning-budget", "0",
            "--threads", "3",
            "--threads-batch", "6",
            "--batch-size", "4096",
            "--ubatch-size", "2048",
            "--no-cache-prompt",
            "--cache-ram", "64",
            "--cache-reuse", "0",
            "--no-mmap",
            "--flash-attn", "on"
        ]

        if self.verbose:
            cmd.append("--verbose")
            print("    [Verbose] Server logs enabled.")
        else:
            cmd.append("--log-disable")

        # Windows-specific: Use CREATE_NEW_PROCESS_GROUP to allow sending signals safely
        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self.server_process = subprocess.Popen(cmd, env=env, creationflags=creationflags)
        
        print(">>> Waiting for model and vision encoder to load...")
        max_retries = 60
        for i in range(max_retries):
            if self._is_server_ready():
                print(">>> Server ready!")
                time.sleep(2)
                return
            if self.server_process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.server_process.returncode}")
            time.sleep(1)
        
        raise RuntimeError("Failed to start llama-server within timeout.")

    def _stop_server(self):
        if self.server_process and not self.is_external_server:
            print(">>> Stopping llama-server...")
            # Simple and effective termination for both Windows and Linux
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(">>> Force killing llama-server...")
                self.server_process.kill()
            self.server_process = None

    def _image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def analyze_frames(self, frames: List[FrameMetadata], transcription: str, batch_size: int = 0) -> List[FrameMetadata]:
        """
        Analyzes frames in batches.
        If batch_size is 0, it is dynamically calculated based on context window.
        """
        if not self._is_server_ready():
            self._start_server()

        max_context_chars = 1500
        context_text = transcription[:max_context_chars] + "..." if len(transcription) > max_context_chars else transcription
        
        # Calculate dynamic batch size if needed
        if batch_size <= 0:
            # Estimate tokens used by text context
            text_tokens = len(context_text) * self.TOKENS_PER_TEXT_CHAR
            available_tokens = self.CONTEXT_SIZE - self.RESERVE_TOKENS - text_tokens
            
            # Calculate how many images fit
            calc_batch = int(available_tokens / self.TOKENS_PER_IMAGE)
            
            # Clamp to reasonable limits (1 to 48)
            # 48 images fits within the 64k context with margin
            batch_size = max(1, min(calc_batch, 50))
            print(f">>> Auto-selected batch size: {batch_size} (based on {self.CONTEXT_SIZE} ctx)")

        total_frames = len(frames)
        print(f"\n>>> Analyzing {total_frames} frames in batches of {batch_size}...")

        for i in range(0, total_frames, batch_size):
            batch = frames[i : i + batch_size]
            current_batch_index = (i // batch_size) + 1
            total_batches = (total_frames + batch_size - 1) // batch_size
            
            print(f"Processing Batch {current_batch_index}/{total_batches} (Frames {i+1}-{i+len(batch)})...")
            
            content: List[Any] = [
                {"type": "text", "text": f"Audio context (may be auto-generated and contain noise/errors): {context_text}\n\nTask: Analyze these {len(batch)} video frames as a continuous sequence. Describe the events or actions shown. Use the audio context if it makes sense, but rely primarily on the visuals. If there is clearly readable text, include it in the summary."}
            ]
            
            for frame in batch:
                img_b64 = self._image_to_base64(frame.path)
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 8192,
                "chat_template_kwargs": {"enable_thinking": False}
            }

            start_time = time.time()
            try:
                # 240s timeout for large batches
                response = requests.post(self.api_url, json=payload, timeout=240)
                response.raise_for_status()
                data = response.json()
                
                message = data["choices"][0]["message"]
                description = message.get("content", "").strip()
                reasoning = message.get("reasoning_content", "").strip()
                
                if not description and reasoning:
                    description = f"[Thinking only, generation cut off]: {reasoning[:200]}..."
                
                elapsed = time.time() - start_time
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                
                print(f"  -> Processed {len(batch)} frames in {elapsed:.2f}s")
                print(f"  -> Tokens: {prompt_tokens} prompt / {tokens} gen")
                
                if self.verbose:
                    if reasoning:
                        print(f"  -> Reasoning:\n{reasoning}\n")
                    print(f"  -> Description:\n{description}\n")
                else:
                    print(f"  -> Description: {description[:100]}...")
                
                for frame in batch:
                    frame.description = description
                    
            except Exception as e:
                print(f"  -> Error processing batch: {e}")
                for frame in batch:
                    frame.description = "Error during analysis."
                continue
            
        return frames

    def generate_video_summary(self, frames: List[FrameMetadata], transcription: str) -> str:
        """Generates a cohesive summary of the entire video."""
        if not frames:
            return "No video content analyzed."

        print("\n>>> Generating comprehensive video summary...")
        
        unique_events = []
        seen_descriptions = set()
        
        for f in frames:
            if f.description and f.description not in seen_descriptions:
                unique_events.append(f"- [{f.timestamp:.1f}s]: {f.description}")
                seen_descriptions.add(f.description)
        
        events_text = "\n".join(unique_events)
        
        prompt = (
            f"You are a video analyst. Based on the following visual timeline and an audio transcription (which may be auto-generated and contain errors), "
            f"write a comprehensive summary of the video. Rely primarily on the visual timeline if the audio seems nonsensical.\n\n"
            f"=== Visual Timeline ===\n{events_text}\n\n"
            f"=== Audio Transcription ===\n{transcription}\n\n"
            f"=== Task ===\n"
            f"Write a detailed paragraph summarizing the main topic and events of the video."
        )

        payload = {
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Provide the summary directly without outputting long thinking processes."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4096
        }

        try:
            start_time = time.time()
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            message = data["choices"][0]["message"]
            summary = message.get("content", "").strip()
            reasoning = message.get("reasoning_content", "").strip()
            
            if not summary and reasoning:
                summary = f"[Thinking only, generation cut off]: {reasoning[:200]}..."
                
            elapsed = time.time() - start_time
            print(f"Summary generated in {elapsed:.2f}s.")
            
            if self.verbose:
                print(f"\n--- Full Summary ---\n{summary}\n--------------------\n")
                
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Summary generation failed."
