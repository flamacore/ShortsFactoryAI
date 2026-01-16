import cv2
import base64
import json
import os
import logging
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
            
            segments, info = model.transcribe(video_path, beam_size=5)
            
            transcript = []
            for segment in segments:
                transcript.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
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
        output_callback: function(str) -> void, for real-time UI updates
        Returns a list of visual events: [{'timestamp': 5.0, 'description': '...'}, ...]
        """
        logger.info(f"👁️ Starting Visual Analysis for {video_path}")
        
        vision_model = self.config['models']['vision_model']
        chunk_duration = self.config['video']['chunk_duration_seconds']
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        visual_timeline = []
        
        # Analyze one frame every 5 seconds
        step_seconds = 5
        step_frames = int(fps * step_seconds)
        
        current_frame = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process every Nth frame
            if current_frame % step_frames == 0:
                timestamp = current_frame / fps
                
                # Resize for LLaVA optimization (672x672 is standard, but we keep aspect ratio)
                # Just resizing to max dimension 672 to save tokens/speed
                h, w = frame.shape[:2]
                scale = 672 / max(h, w)
                frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
                # Encode to base64
                _, buffer = cv2.imencode('.jpg', frame_resized)
                base64_image = base64.b64encode(buffer).decode('utf-8')
                
                # Temp save frame for Ollama (Python lib supports bytes, but path is safer sometimes)
                # Actually ollama python lib supports bytes in 'images' list but base64 string is also fine usually?
                # The official python lib expects path OR base64 bytes. Let's send bytes to `ollama_utils`
                # buffer.tobytes() might be what we need if passing direct bytes
                
                # Let's write to a temp file to be 100% robust with Ollama lib quirks
                temp_frame_path = os.path.join(self.config['folders']['temp'], f"frame_{current_frame}.jpg")
                with open(temp_frame_path, "wb") as f:
                    f.write(buffer)
                
                base_prompt = "Describe this scene briefly. Focus on action, emotion, and key subjects."
                prompt = f"Context: {context_guidance}. {base_prompt}" if context_guidance else base_prompt

                if output_callback:
                    output_callback(f"Analyzing frame at {timestamp:.1f}s...")
                
                description = self.ollama_service.analyze_image(
                    model=vision_model, 
                    image_path=temp_frame_path,
                    prompt=prompt
                )
                
                if output_callback:
                    output_callback(f"⏱️ {timestamp:.1f}s: {description}")
                
                visual_timeline.append({
                    "timestamp": timestamp,
                    "description": description
                })
                
                # Cleanup
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
            
            current_frame += 1
            
            # Simple progress log
            if current_frame % (step_frames * 10) == 0:
                logger.info(f"Processed {current_frame}/{total_frames} frames")

        cap.release()
        
        # Do not unload model here, Director might need Ollama service (although different model)
        # But `ollama_utils` handles the switching.
        
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
