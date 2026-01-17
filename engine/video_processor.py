import os
import logging
import sys
import subprocess
import cv2
from moviepy import VideoFileClip, CompositeVideoClip
from .subtitle_generator import SubtitleGenerator

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, config, output_override=None):
        self.config = config
        # Use override if provided (per-run folder), else default
        self.output_folder = output_override if output_override else config['folders']['output']
        self.temp_folder = config['folders']['temp']
        self.device = config['gpu']['device']
        self.subtitle_gen = SubtitleGenerator(config)

    def create_short(self, video_path, clip_data, transcript, index):
        """
        Renders a single short video using MoviePy and SubtitleGenerator.
        """
        start = clip_data['start_time']
        end = clip_data['end_time']
        duration = end - start
        
        output_filename = f"short_{index+1}_{int(start)}.mp4"
        output_path = os.path.join(self.output_folder, output_filename)
        
        logger.info(f"Rendering Clip {index+1}: {start}-{end}")
        
        # 1. Smart Crop Calculation (Process Isolation)
        crop_mode = self.config.get('video', {}).get('crop_mode', 'face')
        crop_x = self._calculate_crop_x(video_path, start, end, crop_mode)
        
        # 2. Prepare Subtitles
        # Filter words that fall into this clip
        words_in_clip = self._extract_words_for_clip(transcript, start, end)
        
        # 3. MoviePy Composition
        # Load Video
        try:
            with VideoFileClip(video_path) as full_video:
                # Clamp end time to actual video duration
                original_duration = full_video.duration
                if end > original_duration:
                    logger.warning(f"Clip end time {end} exceeds video duration {original_duration}. Clamping.")
                    end = original_duration
                    
                # Recalculate duration
                duration = end - start
                
                # If start is also past duration (rare but possible), skip
                if start >= original_duration:
                    logger.error(f"Clip start time {start} is beyond video duration {original_duration}. Skipping.")
                    return None

                # Subclip
                video = full_video.subclipped(start, end)
                
                # Crop logic
                # Target dimensions (9:16)
                h = video.h
                w = video.w
                target_w = int(h * 9 / 16)
                
                # Ensure crop is within bounds
                if crop_x < 0: crop_x = 0
                if crop_x + target_w > w: crop_x = w - target_w
                
                # Apply Crop
                video = video.cropped(x1=int(crop_x), y1=0, width=target_w, height=h)
                
                # Generate Subtitle Overlay Clips
                # Note: SubtitleGenerator creates ImageClips positioned at center
                # We need to ensure they are relative to the new video.size
                sub_clips = self.subtitle_gen.create_subtitle_clips(words_in_clip, target_w, h)
                
                # Composite
                # Ensure the main video is the bottom layer
                # CompositeVideoClip takes a list of clips
                final_video = CompositeVideoClip([video] + sub_clips)
                
                # Write File
                # Use NVENC if configured, otherwise libx264 (slow but compatible)
                codec = 'libx264'
                ffmpeg_params = []
                
                if self.config['gpu']['use_nvenc']:
                    codec = 'h264_nvenc'
                    # -rc:v vbr_hq -cq:v 19 -b:v 5M ?
                    ffmpeg_params = ['-rc:v', 'vbr_hq', '-cq:v', '19'] 
                    logger.info("Using NVENC encoding")
                
                final_video.write_videofile(
                    output_path,
                    codec=codec,
                    audio_codec='aac',
                    fps=video.fps,
                    threads=4,
                    ffmpeg_params=ffmpeg_params,
                    logger=None # Silence standard moviepy logger to avoid noise, uses 'proglog' usually
                )
                
                logger.info(f"Generated: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Render failed: {e}", exc_info=True)
            raise e

    def _extract_words_for_clip(self, transcript, start, end):
        """
        Flattens the transcript segments into a list of words within the time range.
        Adjusts timestamps to be relative to the clip start.
        """
        words = []
        for segment in transcript:
            seg_start = segment['start']
            seg_end = segment['end']
            
            # Simple overlap check for segment
            if seg_end < start or seg_start > end:
                continue
                
            # If segment has word-level data
            if 'words' in segment and segment['words']:
                for w in segment['words']:
                    w_start = w['start']
                    w_end = w['end']
                    
                    if w_end < start or w_start > end:
                        continue
                        
                    # Relative timing
                    rel_start = max(0, w_start - start)
                    rel_end = min(end - start, w_end - start)
                    
                    if rel_end > rel_start:
                        words.append({
                            'word': w['word'],
                            'start': rel_start,
                            'end': rel_end
                        })
            else:
                # Fallback if no word timestamps (e.g. model didn't support it or failed)
                # Not ideal for "snappy", but we handle it by chunking the full text roughly?
                # No, we just skip for now or implement better fallback later if needed.
                # In current fast-whisper setup, words=None if timestamps=False.
                # We enabled timestamps=True in analysis.py
                pass
                
        return words

    def _calculate_crop_x(self, video_path, start, end, mode='face'):
        """
        Calculates the X position for the 9:16 crop.
        Modes: 'face' (YOLO), 'center' (Simple).
        """
        try:
            # Check dimensions first using OpenCV to avoid moviepy overhead
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            target_width = int(frame_height * 9/16)
            
            if mode == 'center':
                logger.info("Using center crop mode.")
                return (frame_width - target_width) / 2

            # Default: Face Detection
            # Run YOLO in isolated subprocess
            # Use sys.executable to ensure we use the same venv python
            script_path = os.path.join(os.path.dirname(__file__), "yolo_crop_subprocess.py")
            cmd = [sys.executable, script_path, video_path, str(start), str(end)]
            
            logger.info("Spawning isolated YOLO subprocess for smart crop...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"YOLO subprocess failed: {result.stderr}")
                logger.warning("Falling back to center crop.")
                return (frame_width - target_width) / 2
                
            avg_x = float(result.stdout.strip())
            logger.info(f"Smart Crop Target X: {avg_x}")

            # Calculate crop window
            crop_x = int(avg_x - (target_width / 2))
            
            # Clamp bounds
            if crop_x < 0: crop_x = 0
            if crop_x + target_width > frame_width: crop_x = frame_width - target_width
            
            return crop_x
            
        except Exception as e:
            logger.warning(f"Smart crop failed ({e}), using center crop.")
            # Fallback
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return (w - (h * 9/16)) / 2
