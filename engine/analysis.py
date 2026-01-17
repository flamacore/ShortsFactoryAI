import cv2
import base64
import json
import os
import logging
import numpy as np
from faster_whisper import WhisperModel
import torch
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self, config, ollama_service):
        self.config = config
        self.ollama_service = ollama_service
        self.device = self.config['gpu']['device']
        self.whisper_size = self.config['models']['whisper_size']

    def transcribe_video(self, video_path):
        """
        Transcribes the entire video using faster-whisper on GPU.
        Returns a list of segments: [{'start': 0.0, 'end': 5.0, 'text': '...'}, ...]
        """
        logger.info(f"🎧 Starting Audio Transcription for {video_path}")
        
        # STRICT GPU CHECK
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CRITICAL: Whisper requested CUDA but GPU is not available in PyTorch!")

        try:
            # Explicitly clear VRAM before loading Whisper
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

            model = WhisperModel(
                self.whisper_size, 
                device=self.device, 
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            
            segments, info = model.transcribe(video_path, beam_size=5, word_timestamps=True)
            
            transcript = []
            for segment in segments:
                words = []
                if segment.words:
                    for w in segment.words:
                        words.append({
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability
                        })

                transcript.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": words
                })
            
            # Unload Whisper to free VRAM
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"✅ Transcription complete. Found {len(transcript)} segments.")
            return transcript
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return []

    def analyze_visuals(self, video_path, output_callback=None, context_guidance=None):
        """
        Extracts frames and uses Ollama Vision model to describe them.
        Resilient: Checks for existing partial checkpoint to resume work.
        """
        logger.info(f"👁️ Starting Visual Analysis for {video_path}")
        
        vision_model = self.config['models']['vision_model']
        # Default step is 5s, but for long videos we might want to increase it.
        # Check if user passed an override in config or use default
        step_seconds = self.config.get('video', {}).get('analysis_interval', 5)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Checkpoint File
        checkpoint_path = f"{video_path}.visual_checkpoint.json"
        visual_timeline = []
        start_timestamp = 0.0
        
        # Load Checkpoint if exists
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding='utf-8') as f:
                    visual_timeline = json.load(f)
                if visual_timeline:
                    start_timestamp = visual_timeline[-1]['timestamp']
                    logger.info(f"Resuming analysis from timestamp: {start_timestamp}s")
                    if output_callback:
                        output_callback(f"♻️ Resuming analysis from {start_timestamp:.1f}s...")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        step_frames = int(fps * step_seconds)
        
        # Fast forward to resume point
        # We can set CAP_PROP_POS_MSEC or just calc frame index
        resume_frame = int(start_timestamp * fps)
        # Verify valid resume frame
        if resume_frame >= total_frames:
             pass # Already done?
        else:
             cap.set(cv2.CAP_PROP_POS_FRAMES, resume_frame)
             
        # Correction: If we just finished timestamp X, we should start at X + step
        # But loop logic below will handle the increment. 
        # Actually simplest to just iterate through intended timestamps.
        
        current_frame = resume_frame
        
        # Determine next target frame
        # We want frames at 0, step, 2*step...
        # If we resumed at T, the next target should be the next multiple of step_frames > resume_frame
        # Or if resume_frame aligned with step, we need the NEXT one.
        
        # If visual_timeline is empty, we start at 0.
        # If visual_timeline has entry for 0.0, 5.0...
        # We need to start at last_entry + step.
        
        while cap.isOpened():
            # Calculate next target timestamp we need
            next_target_time = 0.0
            if visual_timeline:
                next_target_time = visual_timeline[-1]['timestamp'] + step_seconds
            else:
                next_target_time = 0.0

            # Convert to frame
            next_target_frame = int(next_target_time * fps)
            
            if next_target_frame >= total_frames:
                break
                
            # Seek if we are far behind (optimization)
            if abs(current_frame - next_target_frame) > 100:
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_target_frame)
                current_frame = next_target_frame

            ret, frame = cap.read()
            if not ret:
                break
            
        # Prepare for Scene Detection
        last_analyzed_tensor = None
        max_static_interval = 20.0 # Force re-analysis if X seconds passed
        last_analyzed_time = -999.0
        
        while cap.isOpened():
            # Calculate next target timestamp we need
            next_target_time = 0.0
            if visual_timeline:
                # Basic interval
                next_target_time = visual_timeline[-1]['timestamp'] + step_seconds
            else:
                next_target_time = 0.0

            # Convert to frame
            next_target_frame = int(next_target_time * fps)
            
            if next_target_frame >= total_frames:
                break
                
            # Seek if we are far behind (optimization)
            if abs(current_frame - next_target_frame) > 100:
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_target_frame)
                current_frame = next_target_frame

            ret, frame = cap.read()
            if not ret:
                break
            
            # Now we are at 'current_frame'. Is it the target?
            # Float comparison safety
            if current_frame >= next_target_frame:
                timestamp = current_frame / fps
                
                # --- SCENE DETECTION OPTIMIZATION (GPU) ---
                # Check if this frame is significantly different from the last ONE we analyzed.
                # If it's static menu drift, we skip calling Ollama, unless max_static_interval passed.
                # We perform this check strictly on GPU using PyTorch tensors.
                
                should_analyze = True
                
                # Upload frame to GPU immediately
                # frame is uint8 (0-255), convert to float for mean calc
                if self.device == 'cuda' or self.device.startswith('cuda'):
                    current_tensor = torch.from_numpy(frame).to(self.device).float()
                else:
                    # Fallback for cpu device (shouldn't happen given instructions but safe)
                    current_tensor = torch.from_numpy(frame).float()

                if last_analyzed_tensor is not None:
                    # Calculate Mean Absolute Pixel Difference on GPU
                    # A difference of < 10.0 (out of 255) usually implies static scene with minor compression/noise
                    diff = torch.mean(torch.abs(current_tensor - last_analyzed_tensor)).item()
                    
                    time_diff = timestamp - last_analyzed_time
                    
                    if diff < 15.0 and time_diff < max_static_interval:
                        should_analyze = False
                        
                        description = f"[STATIC SCENE] {visual_timeline[-1]['description']}"
                        # We add it to timeline so next target moves forward
                        
                        entry = {
                            "timestamp": timestamp,
                            "description": description,
                            "is_static": True
                        }
                        visual_timeline.append(entry)
                        
                        # Save checkpoint even on skip
                        with open(checkpoint_path, "w", encoding='utf-8') as f:
                             json.dump(visual_timeline, f, ensure_ascii=False, indent=2)
                        
                        current_frame += step_frames - 1 # Jump ahead!
                        should_analyze = False
                
                if should_analyze:
                    # Resize for LLaVA optimization
                    h, w = frame.shape[:2]
                    scale = 672 / max(h, w)
                    frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    
                    # Encode to base64
                    _, buffer = cv2.imencode('.jpg', frame_resized)
                    
                    temp_frame_path = os.path.join(self.config['folders']['temp'], f"frame_{current_frame}.jpg")
                    with open(temp_frame_path, "wb") as f:
                        f.write(buffer)
                    
                    base_prompt = "Describe this scene briefly. Focus on action, emotion, and key subjects."
                    prompt = f"Context: {context_guidance}. {base_prompt}" if context_guidance else base_prompt

                    if output_callback:
                        output_callback(f"Analyzing frame at {timestamp:.1f}s...")
                    
                    try:
                        description = self.ollama_service.analyze_image(
                            model=vision_model, 
                            image_path=temp_frame_path,
                            prompt=prompt
                        )
                        
                        if output_callback:
                            output_callback(f"⏱️ {timestamp:.1f}s: {description}")
                        
                        entry = {
                            "timestamp": timestamp,
                            "description": description
                        }
                        visual_timeline.append(entry)
                        
                        # SAVE CHECKPOINT
                        with open(checkpoint_path, "w", encoding='utf-8') as f:
                            json.dump(visual_timeline, f, ensure_ascii=False, indent=2)
                        
                        # Update State
                        last_analyzed_tensor = current_tensor
                    
                    except Exception as e:
                        logger.error(f"Failed to analyze frame at {timestamp}: {e}")

                    finally:
                        # Cleanup
                        if os.path.exists(temp_frame_path):
                            os.remove(temp_frame_path)
                    
            current_frame += 1
            
            # Simple progress log
            if current_frame % (step_frames * 10) == 0:
                logger.info(f"Processed {current_frame}/{total_frames} frames")

        cap.release()
        
        return visual_timeline

    def run_full_analysis(self, video_path, status_callback=None, context_guidance=None):
        """
        Orchestrates the full analysis: Audio -> Visuals -> Combined Log
        """
        # 1. Transcribe (Whisper)
        if status_callback: status_callback("🎧 Transcribing Audio...")
        transcript = self.transcribe_video(video_path)
        
        # 2. Analyze Visuals (Ollama)
        if status_callback: status_callback("👁️ Analyzing Visuals...")
        visuals = self.analyze_visuals(video_path, status_callback, context_guidance=context_guidance)
        
        # 3. Combine
        analysis_log = {
            "video_path": video_path,
            "transcript": transcript,
            "visuals": visuals
        }
        
        return analysis_log
