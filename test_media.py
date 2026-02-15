import os
from src.media_processor import MediaProcessor

def test_extraction():
    video_path = "data/input/shudaki_-20251114-0001_mp4_h265-crf30-medium.mp4"
    temp_dir = "data/temp_test"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    print(f"--- Testing Hybrid MediaProcessor with: {video_path} ---")
    processor = MediaProcessor(temp_dir=temp_dir)
    
    try:
        # 1. Check duration
        duration = processor.get_video_duration(video_path)
        print(f"Success: Video duration is {duration:.2f}s")
        
        # 2. Extract frames using hybrid method
        print("Extracting frames (Scene threshold 0.1 OR max interval 30s)...")
        # Ensure the directory is clean for the new test
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        frames = processor.extract_frames(video_path, scene_threshold=0.1, max_interval=30.0)
        print(f"Success: Extracted {len(frames)} frames.")
        
        for i, frame in enumerate(frames):
            print(f"  Frame {i+1:02d}: {frame.path} at {frame.timestamp:.2f}s")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    test_extraction()
