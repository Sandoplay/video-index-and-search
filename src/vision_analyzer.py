import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from typing import List, Optional
from .models import FrameMetadata

class VisionAnalyzer:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct", load_in_4bit: bool = True):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is None:
            print(f"Loading VLM model: {self.model_id}...")
            
            bnb_config = None
            if self.load_in_4bit and self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            # Optimization 1: Lower resolution limit (approx 360x360 to 480x480 range)
            # This drastically reduces activation memory
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                min_pixels=256 * 28 * 28,
                max_pixels=320 * 28 * 28 
            )
            
            # Optimization 2: Use SDPA (PyTorch's native efficient attention)
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                quantization_config=bnb_config,
                device_map={"": 0} if self.device == "cuda" else None,
                trust_remote_code=True,
                attn_implementation="sdpa" # Much more memory efficient than 'eager'
            )

    def analyze_frames(self, frames: List[FrameMetadata], transcription: str) -> List[FrameMetadata]:
        """Analyzes each frame with the provided transcription context."""
        self._load_model()
        
        # Aggressive context truncation
        max_context_chars = 1500 
        context = transcription[:max_context_chars] + "..." if len(transcription) > max_context_chars else transcription

        for frame in frames:
            # Clear cache BEFORE processing to ensure max space for activations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            image = Image.open(frame.path).convert("RGB")
            
            prompt_text = f"Context from audio: {context}\n\nTask: Describe what is happening in this frame."
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self._processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Optimization 3: use_cache=False saves VRAM (no KV cache) at cost of slightly slower generation
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs, 
                    max_new_tokens=128,
                    use_cache=False 
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            frame.description = response.strip()
            print(f"[{frame.timestamp:.2f}s] Analyzed: {frame.description[:80]}...")
            
        return frames
