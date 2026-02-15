import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import List, Optional
from .models import FrameMetadata

class VisionAnalyzer:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct", load_in_4bit: bool = False):
        self.model_id = model_id
        # We ignore load_in_4bit default to force native speed for small models
        self.load_in_4bit = False 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is None:
            print(f"Loading VLM model: {self.model_id} (Native bfloat16)...")
            
            # Using native bfloat16 without quantization for maximum speed
            # The 2B model takes only ~4.5GB VRAM in fp16/bf16
            
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                min_pixels=128 * 28 * 28, 
                max_pixels=256 * 28 * 28  
            )
            
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map={"": 0} if self.device == "cuda" else None,
                trust_remote_code=True,
                attn_implementation="sdpa"
            )

    def analyze_frames(self, frames: List[FrameMetadata], transcription: str, batch_size: int = 4) -> List[FrameMetadata]:
        """
        Analyzes frames in batches to provide video-level context and speed up processing.
        """
        self._load_model()
        
        # Prepare context
        max_context_chars = 1500 
        context = transcription[:max_context_chars] + "..." if len(transcription) > max_context_chars else transcription

        # Process in chunks
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            
            # Clean memory before batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Processing segment {i//batch_size + 1} (Frames {i+1}-{i+len(batch)})...")

            # Load images
            images = [Image.open(f.path).convert("RGB") for f in batch]
            
            # Formulate prompt for a SEQUENCE of images
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            
            # Prompt asking for a continuous description
            prompt_text = (
                f"Context from audio transcription: {context}\n\n"
                f"These are {len(images)} consecutive frames from a video clip. "
                "Describe the flow of events and actions shown across these frames as a coherent story. "
                "Focus on what is happening and how the scene evolves."
            )
            content.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": content}]
            
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self._processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Generate batch description
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs, 
                    max_new_tokens=256,
                    use_cache=False 
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            description = response.strip()
            print(f"Segment Description: {description[:100]}...\n")

            for f in batch:
                f.description = description
            
        return frames
