import torch
import time
import os
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def profile_generation():
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    frames_dir = "data/temp_test"
    # Create fake frames if not exist for testing batching
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        # Assuming there is at least one image somewhere or we create dummy ones
    
    # Check for existing frames
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if len(frame_files) < 4:
        print("Not enough frames for batch test. Please run test_media.py first.")
        return
    
    batch_files = frame_files[:4]
    print(f"Profiling batch of {len(batch_files)} images...")

    # 1. Load Model
    start_time = time.time()
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        min_pixels=128*28*28, 
        max_pixels=256*28*28
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    print(f"Model load time: {time.time() - start_time:.2f}s")

    # 2. Load Images
    start_time = time.time()
    images = [Image.open(f).convert("RGB") for f in batch_files]
    print(f"Image load time: {time.time() - start_time:.4f}s")

    # 3. Preprocessing
    start_time = time.time()
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images] + 
                       [{"type": "text", "text": "Describe these images."}]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    print(f"Preprocessing time (CPU): {time.time() - start_time:.4f}s")

    # 4. Move to GPU
    start_time = time.time()
    inputs = inputs.to("cuda")
    print(f"Move to GPU time: {time.time() - start_time:.4f}s")
    
    # Print input stats
    print(f"Input tokens shape: {inputs.input_ids.shape}")
    print(f"Visual tokens per image (approx): {(inputs.input_ids.shape[1] - 50) / 4}")

    # 5. Warmup
    print("Warming up GPU...")
    # warm up helps to stabilize CUDA kernels
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    torch.cuda.synchronize() # Wait for all kernels

    # 6. Generation
    print("Starting generation...")
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, use_cache=True) # Try use_cache=True for speed
    torch.cuda.synchronize()
    gen_time = time.time() - start_time
    
    num_tokens = generated_ids.shape[1] - inputs.input_ids.shape[1]
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Tokens generated: {num_tokens}")
    print(f"Speed: {num_tokens / gen_time:.2f} tokens/s")

if __name__ == "__main__":
    profile_generation()
