
import logging
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"Torch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

try:
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    print("Moving to CUDA...")
    model.to('cuda')
    print("Running dummy inference...")
    # Create a dummy image
    img = torch.zeros((1, 3, 640, 640), device='cuda')
    results = model(img)
    print("Success!")
except Exception as e:
    print(f"CRASHED: {e}")
    import traceback
    traceback.print_exc()
