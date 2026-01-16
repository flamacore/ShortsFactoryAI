import json
import re
import logging

logger = logging.getLogger(__name__)

class Director:
    def __init__(self, config, ollama_service):
        self.config = config
        self.ollama_service = ollama_service
    
    def select_viral_clips(self, analysis_log, status_callback=None, context_guidance=None):
        """
        Uses an LLM to analyze the timeline and pick the best segments.
        """
        context_model = self.config['models']['context_model']
        max_shorts = self.config['video']['max_shorts']
        
        # 1. Prepare the Context Timeline
        timeline_text = self._create_timeline_prompt(analysis_log)
        
        if status_callback:
            status_callback("🧠 Director is thinking...")
            status_callback(f"Timeline Length: {len(timeline_text)} chars")
        
        # 2. Construct Prompt
        system_prompt = (
            "You are a viral content expert for YouTube Shorts. "
            "Your goal is to select the most engaging segments from a video timeline. "
            "Segments must be self-contained, funny, interesting, or high-energy. "
            "Strictly output JSON only."
        )

        guidance_text = f"Additional User Context for this video: '{context_guidance}'. Use this to identify relevant segments." if context_guidance else ""
        
        user_prompt = (
            f"Here is the timeline of a video (Visuals + Audio):\n\n"
            f"{timeline_text}\n\n"
            f"{guidance_text}\n\n"
            f"Identify exactly {max_shorts} candidate segments that would make viral Shorts (duration 15s to 60s). "
            "Return a JSON object with a key 'clips' containing a list of objects. "
            "Each object must have: 'start_time' (float), 'end_time' (float), 'score' (1-10), 'reason' (string), 'title' (string). "
            "\n\nOutput JSON ONLY. No markdown formatting."
        )
        
        # 3. Query LLM
        response_text = self.ollama_service.generate_text(
            model=context_model,
            prompt=f"{system_prompt}\n\n{user_prompt}"
        )
        
        if status_callback:
            status_callback(f"Director Output:\n{response_text[:500]}...")
            
        # 4. Parse JSON
        clips = self._parse_json_response(response_text)
        
        # 5. Validate & Filter
        valid_clips = self._validate_clips(clips, analysis_log)
        
        return valid_clips, response_text

    def _create_timeline_prompt(self, log):
        """
        Merges visual and audio logs into a readable text format.
        """
        visuals = log.get('visuals', [])
        transcript = log.get('transcript', [])
        
        # Simple merge sort or just bucket by time
        # Let's create a text stream
        
        events = []
        for v in visuals:
            events.append({
                'time': v['timestamp'],
                'type': 'VISUAL',
                'content': v['description']
            })
        for t in transcript:
            events.append({
                'time': t['start'],
                'type': 'AUDIO',
                'content': t['text']
            })
            
        events.sort(key=lambda x: x['time'])
        
        text_lines = []
        for e in events:
            time_str = self._format_time(e['time'])
            line = f"[{time_str}] ({e['type']}) {e['content']}"
            text_lines.append(line)
            
        return "\n".join(text_lines)

    def _format_time(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    def _parse_json_response(self, text):
        """
        Robust JSON parsing from LLM output.
        """
        # 1. Strip thinking blocks from DeepSeek/R1 if present
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        # 2. Extract content from ```json ... ``` tags
        # Use simple find logic to avoid regex recursion issues with nested braces
        json_marker = "```json"
        if json_marker in text:
            start = text.find(json_marker) + len(json_marker)
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        
        # 3. Find outermost braces
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON Parse Error: {e}")
        
        logger.error(f"Failed to parse JSON from Director output: {text[:200]}...")
        return {"clips": []}

    def _validate_clips(self, data, log):
        """
        Ensures clips are within bounds and valid.
        """
        clips = data.get('clips', [])
        if not clips:
            return []
            
        valid = []
        for c in clips:
            # Ensure float types
            try:
                start = float(c.get('start_time', 0))
                end = float(c.get('end_time', 0))
                if end - start < 5: continue # Too short
                valid.append(c)
            except:
                continue
        
        # Sort by score
        valid.sort(key=lambda x: x.get('score', 0), reverse=True)
        return valid[:self.config['video']['max_shorts']]
