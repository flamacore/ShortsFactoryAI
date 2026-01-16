
import logging
import torch
from ultralytics import YOLO
from faster_whisper import WhisperModel
import gc
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

def test_conflict():
    print("--- 1. Init Whisper (CTranslate2) ---")
    try:
        w_model = WhisperModel("tiny", device="cuda", compute_type="float16")
        print("Whisper loaded.")
        # Do a small transcribe
        # checking if we need a file. let's just create a dummy if easier or skip.
        # Just loading might be enough to pollute DLL space.
        del w_model
        gc.collect()
        torch.cuda.empty_cache()
        print("Whisper unloaded.")
    except Exception as e:
        print(f"Whisper Init Failed: {e}")
        return

    print("--- 2. Init YOLO (Torch) ---")
    try:
        y_model = YOLO('yolov8n.pt')
        y_model.to('cuda')
        print("YOLO loaded on CUDA.")
        
        # Simulate passing a CV2 frame (CPU numpy)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        print("Running inference...")
        res = y_model(frame)
        print("Success!")
    except Exception as e:
        print(f"YOLO Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conflict()
