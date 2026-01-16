import ffmpeg
import os
import logging
# from ultralytics import YOLO # Removed to avoid contamination
import cv2
import math
import sys
# import torch  # Removed to avoid contamination

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.output_folder = config['folders']['output']
        self.temp_folder = config['folders']['temp']
        self.device = config['gpu']['device']
        
        # Load Face Model (lazy load)
        self.face_model = None

    def _load_face_model(self):
        if not self.face_model:
            logger.info("Loading YOLOv8 Model...")
            # Use standard yolov8n.pt which handles 'person' class well and auto-downloads
            self.face_model = YOLO('yolov8n.pt') 

    def create_short(self, video_path, clip_data, transcript, index):
        """
        Renders a single short video using FFmpeg and NVENC.
        """
        start = clip_data['start_time']
        end = clip_data['end_time']
        duration = end - start
        
        output_filename = f"short_{index+1}_{int(start)}.mp4"
        output_path = os.path.join(self.output_folder, output_filename)
        
        logger.info(f"Rendering Clip {index+1}: {start}-{end}")
        
        # 1. Calculate Crop (Smart Static Crop)
        crop_x = self._calculate_smart_crop(video_path, start, end)
        
        # 2. Generate Subtitles (SRT)
        srt_path = self._generate_srt(transcript, start, end, index)
        
        # 3. Build FFmpeg Command
        self._render_ffmpeg(video_path, start, duration, crop_x, srt_path, output_path)
        
        return output_path

    def _calculate_smart_crop(self, video_path, start, end):
        """
        Analyzes the segment to find the average X position of the main face.
        Returns the x value for the top-left corner of the 9:16 crop.
        """
        import subprocess
        
        try:
            # Check dimensions first
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Run YOLO in isolated subprocess to avoid DLL conflicts with Faster-Whisper
            script_path = os.path.join(os.path.dirname(__file__), "yolo_crop_subprocess.py")
            cmd = [sys.executable, script_path, video_path, str(start), str(end)]
            
            logger.info("Spawning isolated YOLO subprocess for smart crop...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"YOLO subprocess failed: {result.stderr}")
                raise RuntimeError("Subprocess crashed")
                
            avg_x = float(result.stdout.strip())
            logger.info(f"Smart Crop Target X: {avg_x}")

            # Calculate crop window
            target_width = int(frame_height * 9/16)
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

    def _generate_srt(self, full_transcript, clip_start, clip_end, index):
        """
        Creates a temporary SRT file for the clip.
        Adjusts timestamps to be relative to clip start.
        """
        subs = []
        for i, seg in enumerate(full_transcript):
            # Check overlap
            # If segment starts after clip ends or ends before clip starts, skip
            if seg['start'] > clip_end or seg['end'] < clip_start:
                continue
                
            # Adjust timings
            rel_start = max(0, seg['start'] - clip_start)
            rel_end = min(clip_end - clip_start, seg['end'] - clip_start)
            
            if rel_end - rel_start < 0.1: continue
            
            subs.append({
                'index': len(subs)+1,
                'start': rel_start,
                'end': rel_end,
                'text': seg['text'].strip()
            })
            
        srt_content = ""
        for s in subs:
            start_fmt = self._format_srt_time(s['start'])
            end_fmt = self._format_srt_time(s['end'])
            srt_content += f"{s['index']}\n{start_fmt} --> {end_fmt}\n{s['text']}\n\n"
            
        srt_path = os.path.abspath(os.path.join(self.temp_folder, f"subs_{index}.srt"))
        
        # Determine content
        if not srt_content.strip():
            # Create a dummy subtitle to prevent 0-byte file error in FFmpeg
            srt_content = "1\n00:00:00,000 --> 00:00:01,000\n...\n\n"

        with open(srt_path, "w", encoding='utf-8') as f:
            f.write(srt_content)
            
        return srt_path
    
    def _format_srt_time(self, seconds):
        # HH:MM:SS,mmm
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _render_ffmpeg(self, video_path, start, duration, crop_x, srt_path, output_path):
        """
        Constructs the FFmpeg command for NVENC encoding.
        """
        # Use relative path for FFmpeg to avoid Windows drive letter colon escaping issues
        # This assumes the script is run from the workspace root
        try:
            srt_relative = os.path.relpath(srt_path, os.getcwd())
            srt_path_ffmpeg = srt_relative.replace('\\', '/')
        except ValueError:
            # Fallback for different drives, though unlikely in this setup
            srt_path_ffmpeg = srt_path.replace('\\', '/').replace(':', '\\\\:')
        
        args = {
            'ss': start,
            't': duration
        }
        
        stream = ffmpeg.input(video_path, **args)
        
        # Probe dimensions
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        height = int(video_stream['height'])
        target_width = int(height * 9/16)
        
        # Audio
        audio = stream.audio
        
        # Video Filter Chain
        # 1. Crop
        # 2. Subtitles
        
        # Note: 'subtitles' filter is usually CPU bound but fast enough. 
        # Using it with NVENC handles the encoding on GPU.
        
        v = stream.video.crop(x=crop_x, y=0, width=target_width, height=height)
        v = v.filter('subtitles', srt_path_ffmpeg, force_style='Fontname=Arial,FontSize=24,PrimaryColour=&H00FFFF,BackColour=&H80000000,BorderStyle=3')
        
        # Output
        # 'h264_nvenc' or 'hevc_nvenc'
        
        try:
            out = ffmpeg.output(
                v, 
                audio, 
                output_path, 
                vcodec='h264_nvenc', 
                video_bitrate='5M',
                acodec='aac'
            )
            out.run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            logger.info(f"Generated: {output_path}")
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg Error: {e.stderr.decode('utf8')}")
            raise e
