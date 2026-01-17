import os
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, CompositeVideoClip, VideoFileClip

logger = logging.getLogger(__name__)

class SubtitleGenerator:
    def __init__(self, config):
        self.config = config
        self.temp_folder = config['folders']['temp']
        # Try to find a nice heavy font
        self.font_path = self._find_font()
        self.font_size = 80  # Base font size for 1080p vertical
        self.max_words = 5

    def _find_font(self):
        # Common windows paths for bold/impact fonts
        candidates = [
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/ariblk.ttf", # Arial Black
            "C:/Windows/Fonts/seguiemj.ttf", # Segoe UI Emoji (fallback)
            "C:/Windows/Fonts/arialbd.ttf"  # Arial Bold
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "arial.ttf" # Last resort PIL default

    def create_subtitle_clips(self, words, video_width, video_height):
        """
        Generates a list of MoviePy clips for the subtitles.
        words: list of dicts {'word': str, 'start': float, 'end': float}
        """
        clips = []
        if not words:
            return []

        # 1. Chunk words
        chunks = self._chunk_words(words)

        # 2. Create compound clips for each chunk
        for chunk in chunks:
            chunk_clips = self._create_karaoke_chunk(chunk, video_width, video_height)
            clips.extend(chunk_clips)

        return clips

    def _chunk_words(self, words):
        chunks = []
        current_chunk = []
        
        for w in words:
            current_chunk.append(w)
            # Max 3-4 words for better viral pacing
            if len(current_chunk) >= 4:
                chunks.append(current_chunk)
                current_chunk = []
                
        if current_chunk:
            chunks.append(current_chunk)
            
        # Ensure continuity
        for i in range(len(chunks) - 1):
            curr = chunks[i]
            next_c = chunks[i+1]
            gap = next_c[0]['start'] - curr[-1]['end']
            if gap < 0.2: 
                curr[-1]['end'] = next_c[0]['start']
                
        return chunks

    def _create_karaoke_chunk(self, chunk_words, width, height):
        """
        Creates a list of clips for a single chunk of text.
        Each word entering triggers a new clip state.
        Uses ACCUMULATION style: "The" -> "The quick" -> "The quick brown"
        And HIGHLIGHTS the current active word.
        """
        clips = []
        full_text = " ".join([w['word'] for w in chunk_words]).strip().upper()
        
        # Determine Base Layout & Font Size for the FULL chunk
        # Use smaller base font (User said "Gigantic" was awful)
        base_font_size = int(width / 12) # ~50px for 600w
        font, lines, line_heights = self._calculate_layout(full_text, width, base_font_size)
        
        # Iterate through words to create states
        # A state lasts from word.start to word.end (or next word start)
        
        accumulated_words = []
        
        for i, active_word in enumerate(chunk_words):
            accumulated_words.append(active_word)
            
            # Duration of this state
            start_t = active_word['start']
            
            # End time is either start of next word, or end of this word if it's the last one
            if i < len(chunk_words) - 1:
                end_t = chunk_words[i+1]['start']
            else:
                end_t = active_word['end']
            
            duration = end_t - start_t
            if duration < 0.05: duration = 0.05

            # Highlight logic: Active word is Yellow, others are White
            # We need to map 'accumulated_words' to the 'lines' structure we calculated
            
            img = self._render_karaoke_frame(
                lines, 
                width, 
                height, 
                font, 
                line_heights, 
                active_word_idx=i, # The index in the full chunk provided
                total_chunk_words=chunk_words # To know what text is what
            )
            
            img_np = np.array(img)
            
            # Handle alpha
            if img_np.shape[2] == 4:
                rgb = img_np[:,:,:3]
                alpha = img_np[:,:,3] / 255.0
                base_clip = ImageClip(rgb).with_start(start_t).with_duration(duration)
                mask_clip = ImageClip(alpha, is_mask=True).with_start(start_t).with_duration(duration)
                clip = base_clip.with_mask(mask_clip)
            else:
                clip = ImageClip(img_np).with_start(start_t).with_duration(duration)

            clip = clip.with_position(('center', 'center'))
            clips.append(clip)
            
        return clips

    def _calculate_layout(self, text, width, start_font_size):
        """
        Calculates the layout (wrapped lines) that fits the constraints.
        Returns: (font_object, lines_list, line_heights_list)
        """
        margin = int(width * 0.1)
        max_w = width - (margin * 2)
        font_size = start_font_size
        
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        
        while True:
            font = self._load_font(font_size)
            lines = []
            current_line = []
            words = text.split()
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                bbox = draw.textbbox((0, 0), test_line, font=font)
                w = bbox[2] - bbox[0]
                if w <= max_w:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
                        current_line = []
            if current_line:
                lines.append(' '.join(current_line))

            # Shrink check
            # 1. Overflow check
            max_line_w = 0
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                if (bbox[2] - bbox[0]) > max_line_w: max_line_w = bbox[2] - bbox[0]
            
            # 2. Density check (User requirement: ~3 words fit)
            fits_width = max_line_w <= max_w
            
            # Calculate avg density
            avg_words = len(words) / max(1, len(lines))
            density_ok = avg_words >= 2.0 # Allow 2 words min per line
            
            if fits_width and (density_ok or font_size <= 30 or len(words) < 3):
                # Calculate heights
                heights = []
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    heights.append(bbox[3] - bbox[1])
                return font, lines, heights
            
            font_size -= 4
            if font_size < 20: 
                # Safety break
                heights = []
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    heights.append(bbox[3] - bbox[1])
                return font, lines, heights

    def _render_karaoke_frame(self, lines, width, height, font, line_heights, active_word_idx, total_chunk_words):
        """
        Renders the lines, but only draws words up to active_word_idx.
        Current active word is YELLOW + Big Stroke.
        Previous words are WHITE + Medium Stroke.
        Future words are INVISIBLE.
        """
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        line_spacing = 15
        total_h = sum(line_heights) + (len(lines) - 1) * line_spacing
        start_y = (height * 0.70) - (total_h / 2)
        
        # Flatten lines back to words to map simple index
        # We need to reconstruct which word belongs to which line and position
        # This is tricky because wrapping logic lost the exact word indices.
        # Simple approach: Re-flow the words into the lines to match positions.
        
        word_counter = 0
        current_y = start_y
        
        stroke_width = max(3, int(font.size / 15))
        
        for i, line in enumerate(lines):
            line_words = line.split()
            
            # Calculate total width of line to center it
            # We need to draw word by word to color individually
            # So we need exact x-positions of each word
            
            # 1. Measure total line width first
            full_line_bbox = draw.textbbox((0, 0), line, font=font)
            line_w = full_line_bbox[2] - full_line_bbox[0]
            start_x = (width - line_w) / 2
            
            cursor_x = start_x
            
            for word in line_words:
                # Check state
                if word_counter > active_word_idx:
                    # Future word: Invisible, but advance cursor
                    pass
                else:
                    # Visible
                    is_active = (word_counter == active_word_idx)
                    
                    fill_color = "#FFD700" if is_active else "white" # Gold for active
                    s_width = stroke_width + 2 if is_active else stroke_width
                    
                    # Draw One Word
                    draw.text((cursor_x, current_y), word, font=font, fill="black", stroke_width=s_width, stroke_fill="black")
                    draw.text((cursor_x, current_y), word, font=font, fill=fill_color, stroke_width=0)
                
                # Advance cursor including space
                w_bbox = draw.textbbox((0, 0), word + " ", font=font)
                cursor_x += (w_bbox[2] - w_bbox[0])
                
                word_counter += 1
                
            current_y += line_heights[i] + line_spacing
            
        return img

    def _load_font(self, size):
        try:
            return ImageFont.truetype(self.font_path, size)
        except:
            return ImageFont.load_default()
