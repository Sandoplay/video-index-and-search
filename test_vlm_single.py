import torch
import os
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

def test_single_frame():
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    # Take the first frame from the processed folder or temp_test
    image_path = "data/temp_test/frame_0001.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found. Please run test_media.py first.")
        return

    print(f"Initial VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    print(f"Loading model {model_id} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, max_pixels=512*28*28)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    print(f"VRAM after loading model: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"VRAM reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")

    print(f"VRAM after processing inputs: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    print("Generating description...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    print(f"\nResponse: {response}")
    print(f"\nFinal VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Final VRAM reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    test_single_frame()
