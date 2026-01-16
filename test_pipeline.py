import os
import sys
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure engine is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.ollama_utils import OllamaService
from engine.analysis import VideoAnalyzer
from engine.director import Director
from engine.video_processor import VideoProcessor

TEST_VIDEO = r"G:\VideoRecords\2026-01-01_14-11-05.mp4"

def test_pipeline():
    logger.info("Starting Full Pipeline Test")

    # 1. Load Config
    if not os.path.exists("config.yaml"):
        logger.error("Config file missing!")
        return
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    # FORCE GPU SETTINGS as per user request (no fallbacks)
    config['gpu']['device'] = 'cuda'
    config['gpu']['use_nvenc'] = True
    
    # Check Video
    if not os.path.exists(TEST_VIDEO):
        logger.error(f"Test video not found at: {TEST_VIDEO}")
        return

    # 2. Initialize Services
    try:
        ollama_svc = OllamaService()
        # Verify Ollama is up
        models = ollama_svc.list_local_models()
        logger.info(f"Ollama Service: ONLINE. Found {len(models)} models.")
    except Exception as e:
        logger.error(f"Ollama Service Failed: {e}")
        return

    # 3. Setup Components
    # Force specific models for test if not in config
    # config['models']['vision_model'] = 'llava' 
    # config['models']['context_model'] = 'llama3'
    
    analyzer = VideoAnalyzer(config, ollama_svc)
    director = Director(config, ollama_svc)
    processor = VideoProcessor(config)

    # 4. Run Analysis
    logger.info("--- Step 1: Analysis ---")
    analysis_log = analyzer.run_full_analysis(TEST_VIDEO, status_callback=lambda x: logger.info(f"Callback: {x}"))
    
    # MOCK TRANSCRIPT (Unconditional overwrite for validation)
    # The silenced video might produce garbage or empty frames. We want to test the subtitle burn-in.
    logger.info("⚠️ Forcing MOCK TRANSCRIPT for validation (Overwriting Whisper output)...")
    analysis_log['transcript'] = [
        {'start': 0.5, 'end': 3.0, 'text': "Let's go! Rushing the portal!"},
        {'start': 3.5, 'end': 6.0, 'text': "Wave 2 incoming, stay sharp."},
        {'start': 7.0, 'end': 10.0, 'text': "This is intense gameplay."},
        {'start': 11.0, 'end': 14.0, 'text': "We need to secure the objective now!"}
    ]

    # 5. Director
    logger.info("--- Step 2: Director ---")
    clips, rationale = director.select_viral_clips(analysis_log, status_callback=lambda x: logger.info(f"Callback: {x}"))
    logger.info(f"Director found {len(clips)} clips")
    
    if not clips:
        logger.error("No clips selected! Test Failed.")
        # Create a dummy clip to force render test if analysis fails purely on content judgement
        # logger.info("Generating dummy clip for render test...")
        # clips = [{'start_time': 10.0, 'end_time': 20.0, 'title': 'Test Clip', 'score': 10, 'reason': 'Test'}]
        # Un-commenting above allows testing renderer even if LLM yields nothing, but user said "no fallbacks"
        raise Exception("Director returned 0 valid clips.")

    # 6. Render
    logger.info("--- Step 3: Rendering ---")
    output_files = []
    
    # Create output dir if needed
    if not os.path.exists(config['folders']['output']):
        os.makedirs(config['folders']['output'])
        
    for i, clip in enumerate(clips):
        logger.info(f"Rendering Clip {i}")
        try:
            out = processor.create_short(TEST_VIDEO, clip, analysis_log['transcript'], i)
            output_files.append(out)
            logger.info(f"Success: {out}")
        except Exception as e:
            logger.error(f"Rendering failed for clip {i}: {e}")
            raise e # No graceful fail

    logger.info("Test Run Complete. Outputs:")
    for f in output_files:
        logger.info(f"- {f}")

if __name__ == "__main__":
    test_pipeline()
