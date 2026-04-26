import os
import base64
import time
import subprocess
import requests
import json
import signal
import sys
from typing import List, Optional, Dict, Any, Tuple
from .models import FrameMetadata, SegmentMetadata

class VisionAnalyzer:
    # Constants based on Qwen3.5-VL and user testing on RTX 5060 Ti
    CONTEXT_SIZE = 64000 # Optimized for 16GB VRAM with Q8_0 quantization
    TOKENS_PER_IMAGE = 1024  # Adjusted for Qwen3.5-VL typical usage
    TOKENS_PER_TEXT_CHAR = 0.3 # Rough estimate
    RESERVE_TOKENS = 2000    # Increased for reasoning/thinking margin
    STRUCTURED_ANALYSIS_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string"
            },
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_frame": {
                            "type": "integer",
                            "minimum": 0
                        },
                        "end_frame": {
                            "type": "integer",
                            "minimum": 0
                        },
                        "description": {
                            "type": "string"
                        }
                    },
                    "required": ["start_frame", "end_frame", "description"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["summary", "segments"],
        "additionalProperties": False
    }

    def __init__(self, model_id: str = "models/Qwen3.5-9B-Q8_0.gguf", load_in_4bit: bool = False, verbose: bool = False, parallel_slots: int = 2):
        self.model_path = model_id
        self.mmproj_path = "models/mmproj-F16.gguf"
        self.server_process = None
        self.api_url = "http://127.0.0.1:8080/v1/chat/completions"
        self.torch_lib = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages", "torch", "lib")
        self.is_external_server = False
        self.verbose = verbose
        self.parallel_slots = parallel_slots

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
        env["LLAMA_SET_ROWS"] = "1"
        
        cmd = [
            server_exe,
            "--model", actual_model_path,
            "--mmproj", actual_mmproj_path,
            "--n-gpu-layers", "-1",
            "--port", "8080",
            "--ctx-size", str(self.CONTEXT_SIZE),
            "--n-predict", "10240",
            "--reasoning-budget", "0",
            "--threads", "8",
            "--threads-batch", "8",
            "--batch-size", "4096",
            "--ubatch-size", "2048",
            "--no-cache-prompt",
            "--cache-ram", "256",
            "--cache-reuse", "0",
            "--no-mmap",
            "--flash-attn", "on",
            "--parallel", str(self.parallel_slots)

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

    def _extract_json_candidate(self, text: str) -> str:
        """Extract a JSON object from plain text or a markdown code fence."""
        candidate = text.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start:end + 1]
        return candidate

    def _parse_structured_response(self, text: str) -> Dict[str, Any]:
        candidate = self._extract_json_candidate(text)
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("Structured response must be a JSON object.")
        return parsed

    def _repair_structured_response(self, raw_text: str) -> Dict[str, Any]:
        repair_prompt = (
            "Convert the previous response into valid JSON matching this schema exactly. "
            "Return ONLY valid JSON, no markdown and no explanations.\n\n"
            "{\n"
            '  "summary": "string",\n'
            '  "segments": [\n'
            '    {"start_frame": 0, "end_frame": 0, "description": "string"}\n'
            "  ]\n"
            "}\n\n"
            f"Previous response:\n{raw_text}"
        )

        payload = {
            "messages": [
                {"role": "user", "content": repair_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
            "json_schema": self.STRUCTURED_ANALYSIS_SCHEMA,
            "chat_template_kwargs": {"enable_thinking": False}
        }

        response = requests.post(self.api_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        repaired = message.get("content", "").strip()
        if not repaired and message.get("reasoning_content"):
            repaired = message.get("reasoning_content", "").strip()
        return self._parse_structured_response(repaired)

    def _fallback_segments(self, frames: List[FrameMetadata], raw_text: str = "") -> Tuple[str, List[SegmentMetadata]]:
        if not frames:
            return "Structured analysis failed.", []

        description = raw_text.strip() or "Error during structured analysis."
        return (
            "Structured analysis failed.",
            [
                SegmentMetadata(
                    start_frame=0,
                    end_frame=len(frames) - 1,
                    start_time=float(frames[0].timestamp),
                    end_time=float(frames[-1].timestamp),
                    description=description,
                )
            ],
        )

    def _normalize_structured_data(self, data: Dict[str, Any], frames: List[FrameMetadata]) -> Tuple[str, List[SegmentMetadata]]:
        if not frames:
            return str(data.get("summary", "No video content analyzed.")).strip(), []

        summary = str(data.get("summary", "")).strip() or "No summary generated."
        raw_segments = data.get("segments", [])
        if not isinstance(raw_segments, list):
            raw_segments = []

        segments: List[SegmentMetadata] = []
        max_index = len(frames) - 1

        for item in raw_segments:
            if not isinstance(item, dict):
                continue

            try:
                start_frame = int(item.get("start_frame", 0))
                end_frame = int(item.get("end_frame", start_frame))
            except (TypeError, ValueError):
                continue

            start_frame = max(0, min(start_frame, max_index))
            end_frame = max(0, min(end_frame, max_index))
            if end_frame < start_frame:
                start_frame, end_frame = end_frame, start_frame

            description = str(item.get("description", "")).strip()
            if not description:
                continue

            segments.append(
                SegmentMetadata(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=float(frames[start_frame].timestamp),
                    end_time=float(frames[end_frame].timestamp),
                    description=description,
                )
            )

        if not segments:
            return (
                summary,
                [
                    SegmentMetadata(
                        start_frame=0,
                        end_frame=len(frames) - 1,
                        start_time=float(frames[0].timestamp),
                        end_time=float(frames[-1].timestamp),
                        description=summary,
                    )
                ],
            )

        return summary, segments

    def _response_timings(self, data: Dict[str, Any], client_wall_seconds: float) -> Dict[str, Any]:
        timings = data.get("timings") or {}
        usage = data.get("usage") or {}

        prompt_ms = timings.get("prompt_ms")
        predicted_ms = timings.get("predicted_ms")
        server_total_seconds = None
        if prompt_ms is not None or predicted_ms is not None:
            server_total_seconds = ((prompt_ms or 0.0) + (predicted_ms or 0.0)) / 1000.0

        result: Dict[str, Any] = {
            "vlm_client_wall_seconds": round(client_wall_seconds, 3),
            "vlm_prompt_tokens": usage.get("prompt_tokens", timings.get("prompt_n", 0)),
            "vlm_completion_tokens": usage.get("completion_tokens", timings.get("predicted_n", 0)),
        }

        if prompt_ms is not None:
            result["vlm_server_prompt_seconds"] = round(prompt_ms / 1000.0, 3)
            result["vlm_server_prompt_tokens_per_second"] = timings.get("prompt_per_second")

        if predicted_ms is not None:
            result["vlm_server_generation_seconds"] = round(predicted_ms / 1000.0, 3)
            result["vlm_server_generation_tokens_per_second"] = timings.get("predicted_per_second")

        if server_total_seconds is not None:
            result["vlm_server_total_seconds"] = round(server_total_seconds, 3)
            result["vlm_client_overhead_or_queue_seconds"] = round(max(0.0, client_wall_seconds - server_total_seconds), 3)

        return result

    def analyze_video_structured(self, frames: List[FrameMetadata], transcription: str, batch_size: int = 0) -> Tuple[str, List[SegmentMetadata], Dict[str, Any]]:
        """
        Analyze sampled video frames and return one structured summary plus timeline segments.
        The standard path sends all sampled frames in a single request.
        """
        if not self._is_server_ready():
            self._start_server()

        if not frames:
            return "No video content analyzed.", [], {"vlm_client_wall_seconds": 0.0}

        max_context_chars = 3000
        context_text = transcription[:max_context_chars] + "..." if len(transcription) > max_context_chars else transcription

        if batch_size <= 0:
            text_tokens = len(context_text) * self.TOKENS_PER_TEXT_CHAR
            available_tokens = self.CONTEXT_SIZE - self.RESERVE_TOKENS - text_tokens
            calc_batch = int(available_tokens / self.TOKENS_PER_IMAGE)
            batch_size = max(1, min(calc_batch, 50))
            print(f">>> Auto-selected structured batch size: {batch_size} (based on {self.CONTEXT_SIZE} ctx)")

        if len(frames) > 50:
            print(f">>> Warning: {len(frames)} frames exceed structured limit 50; using the first 50 frames.")
            frames_for_request = frames[:50]
        else:
            frames_for_request = frames

        frame_lines = "\n".join(
            f"- Frame {i}: {frame.timestamp:.2f}s"
            for i, frame in enumerate(frames_for_request)
        )

        instruction = (
            "Audio context may be auto-generated and contain noise/errors. Rely primarily on the visuals when they conflict.\n\n"
            f"Audio context:\n{context_text}\n\n"
            "Frames are provided below in chronological order. Frame indices are zero-based positions in this sampled frame list:\n"
            f"{frame_lines}\n\n"
            "Task: Analyze the video as a continuous sequence and return a compact structured result.\n"
            "Group visually similar consecutive frames into meaningful timeline segments. Do not describe every frame independently unless every frame is genuinely distinct.\n"
            "Use 4-12 segments for a typical video. Include clearly readable on-screen text inside the relevant segment description.\n\n"
            "Return ONLY valid JSON. No markdown. No explanations. Use exactly this schema:\n"
            "{\n"
            '  "summary": "A concise paragraph summarizing the whole video.",\n'
            '  "segments": [\n'
            '    {"start_frame": 0, "end_frame": 0, "description": "What happens in this segment."}\n'
            "  ]\n"
            "}"
        )

        content: List[Any] = [{"type": "text", "text": instruction}]
        for i, frame in enumerate(frames_for_request):
            content.append({"type": "text", "text": f"Frame {i} ({frame.timestamp:.2f}s):"})
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
            "json_schema": self.STRUCTURED_ANALYSIS_SCHEMA,
            "chat_template_kwargs": {"enable_thinking": False}
        }

        print(f"\n>>> Analyzing {len(frames_for_request)} frames with structured JSON output...")
        start_time = time.time()
        raw_response = ""

        try:
            response = requests.post(self.api_url, json=payload, timeout=240)
            response.raise_for_status()
            data = response.json()
            client_wall_seconds = time.time() - start_time

            message = data["choices"][0]["message"]
            raw_response = message.get("content", "").strip()
            reasoning = message.get("reasoning_content", "").strip()
            if not raw_response and reasoning:
                raw_response = reasoning

            try:
                parsed = self._parse_structured_response(raw_response)
            except Exception as parse_error:
                print(f"  -> Structured JSON parse failed: {parse_error}. Retrying repair once...")
                parsed = self._repair_structured_response(raw_response)

            summary, segments = self._normalize_structured_data(parsed, frames_for_request)
            elapsed = client_wall_seconds
            response_timings = self._response_timings(data, client_wall_seconds)
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)

            print(f"  -> Structured analysis client wall: {elapsed:.2f}s")
            if "vlm_server_total_seconds" in response_timings:
                print(
                    f"  -> Server active: {response_timings['vlm_server_total_seconds']:.2f}s "
                    f"(prompt {response_timings.get('vlm_server_prompt_seconds', 0):.2f}s / "
                    f"gen {response_timings.get('vlm_server_generation_seconds', 0):.2f}s), "
                    f"queue/overhead {response_timings['vlm_client_overhead_or_queue_seconds']:.2f}s"
                )
            print(f"  -> Tokens: {prompt_tokens} prompt / {tokens} gen")
            print(f"  -> Segments: {len(segments)}")

            if self.verbose:
                print(f"  -> Raw structured response:\n{raw_response}\n")

            return summary, segments, response_timings
        except Exception as e:
            print(f"  -> Structured analysis failed: {e}")
            summary, segments = self._fallback_segments(frames_for_request, raw_response)
            return summary, segments, {"vlm_client_wall_seconds": round(time.time() - start_time, 3)}

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
