
import sys
import os
import cv2
import logging
# Configure concise logging for the subprocess
logging.basicConfig(level=logging.ERROR)

def get_smart_crop(video_path, start_time, end_time):
    try:
        from ultralytics import YOLO
        import torch
        
        # Explicit GPU check in subprocess
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            # If this fails, the parent process catches the non-zero exit or empty output
            sys.stderr.write("CUDA not available in crop subprocess\n")
            sys.exit(1)

        model = YOLO('yolov8n.pt')
        model.to(device)
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Scan at 1fps to be fast
        step = int(fps)
        duration = end_time - start_time
        if duration <= 0: duration = 5
        
        max_frames = int(duration * fps)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        centers = []
        count = 0
        
        while count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if count % step == 0:
                results = model(frame, verbose=False)
                boxes = results[0].boxes
                # Class 0 is person
                persons = boxes[boxes.cls == 0]
                
                if len(persons) > 0:
                    # Largest person
                    largest = max(persons, key=lambda p: p.xywh[0][2] * p.xywh[0][3])
                    x_center = largest.xywh[0][0].item()
                    centers.append(x_center)
            
            count += 1
            
        cap.release()
        
        if not centers:
            # Return center of frame
            print(frame_width / 2)
        else:
            print(sum(centers) / len(centers))
            
    except Exception as e:
        sys.stderr.write(f"Error in crop subprocess: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(1)
        
    v_path = sys.argv[1]
    s_time = float(sys.argv[2])
    e_time = float(sys.argv[3])
    
    get_smart_crop(v_path, s_time, e_time)
