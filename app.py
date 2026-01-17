import streamlit as st
import yaml
import os
import sys
from datetime import datetime
import logging # Added missing import

# Ensure the root directory is in the path to import engine modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.ollama_utils import OllamaService
from engine.analysis import VideoAnalyzer
from engine.director import Director
from engine.video_processor import VideoProcessor
import json

# Page Layout (Must be first Streamlit command)
st.set_page_config(page_title="Shorts AI Factory", layout="wide", initial_sidebar_state="expanded")

if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- CENTRAL LOGGING SINK ---
# Ensure log dir exists before configuring logging
log_dir = "logs"
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        if 'folders' in config and 'logs' in config['folders']:
            log_dir = config['folders']['logs']
except: pass

if not os.path.exists(log_dir): os.makedirs(log_dir)
session_log_file = os.path.join(log_dir, f"session_{st.session_state.session_id}.log")

# Configure Root Logger to write to file
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout), # CMD
            logging.FileHandler(session_log_file, encoding='utf-8') # FILE
        ],
        force=True # Reset any existing config
    )
except Exception as e:
    print(f"Logging setup failed: {e}")

# Load Config
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("config.yaml not found!")
    st.stop()

# Initialize Backend
if 'ollama_svc' not in st.session_state:
    st.session_state.ollama_svc = OllamaService()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Loading
    local_models = st.session_state.ollama_svc.list_local_models()
    all_model_names = [m.model for m in local_models] if local_models else ["No models found"]
    
    # Smart Filter for Vision
    vision_candidates = st.session_state.ollama_svc.get_vision_models()
    
    vision_model = st.selectbox(
        "👁️ Vision Model", 
        vision_candidates, 
        index=0 if vision_candidates else 0,
        help="Select a model that supports LLaVA/Vision. Auto-detected based on 'families' or name."
    )
    
    context_model = st.selectbox(
        "🧠 Director Model", 
        all_model_names, 
        index=min(1, len(all_model_names)-1) if all_model_names else 0,
        help="The thinking model (Llama3, Mistral)"
    )
    
    max_shorts = st.slider("Max Shorts", 1, 5, config['video']['max_shorts'])
    
    # Cropping Mode
    crop_options = {
        "Face Detection (Smart)": "face",
        "Center Crop (Fast)": "center"
    }
    selected_crop_label = st.selectbox(
        "✂️ Cropping Mode",
        list(crop_options.keys()),
        index=0,
        help="Choose how to crop the landscape video into vertical shorts."
    )
    crop_mode = crop_options[selected_crop_label]

    # Update config in memory (for this session)
    config['models']['vision_model'] = vision_model
    config['models']['context_model'] = context_model
    config['video']['max_shorts'] = max_shorts
    if 'video' not in config: config['video'] = {}
    config['video']['crop_mode'] = crop_mode
    
    st.divider()
    st.markdown("### 🖥️ Hardware Stats")
    st.caption(f"Device: {config['gpu']['device'].upper()}")
    st.caption(f"NVENC: {'Enabled' if config['gpu']['use_nvenc'] else 'Disabled'}")

# --- MAIN PAGE ---
st.title("🎬 Viral Shorts Factory")
st.markdown("Drag, drop, and let the GPU cook.")

uploaded_file = st.file_uploader("Upload Long-form Video", type=['mp4', 'mov', 'mkv'])

if uploaded_file:
    # Save to temp
    input_dir = config['folders']['input']
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        
    temp_path = os.path.join(input_dir, uploaded_file.name)
    if not os.path.exists(temp_path):
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    st.video(temp_path)
    
    # Real-time Console
    with st.expander("🧠 AI Thought Stream", expanded=True):
        console_placeholder = st.empty()
        console_text = []
        
        # Setup run logger path (will be created on first write)
        if not os.path.exists(config['folders']['logs']):
            os.makedirs(config['folders']['logs'])
        
        # Create a session ID for logging if not exists
        if 'session_id' not in st.session_state:
            st.session_state.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
        run_log_path = os.path.join(config['folders']['logs'], f"session_{st.session_state.session_id}.log")

        def update_console(text):
            # 1. UI Update
            console_text.append(text)
            # Keep only last 15 lines in preview
            console_placeholder.markdown(f"```text\n" + "\n".join(console_text[-15:]) + "\n```")
            
            # 2. Console Update (Terminal)
            print(f"[UI] {text}")
            
            # 3. File Update
            try:
                with open(run_log_path, "a", encoding="utf-8") as f:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    f.write(f"[{timestamp}] {text}\n")
            except Exception as e:
                print(f"Failed to write to log: {e}")

        console_placeholder.text("Waiting for activation...")
        if os.path.exists(run_log_path): 
            try:
                # Pre-load existing logs if session continues (or just show last few)
                 pass
            except: pass

    user_guidance = st.text_area("Context / Guidance (Optional)", placeholder="E.g., This is a stream of Destiny: Rising. Ignore the menu screens.", height=68)

    if st.button("🚀 Generate Shorts", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. Init Engines
            status_text.text("Initializing Engines...")
            
            # Create run-specific output folder (separate for each click)
            run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_folder_name = f"run_{run_timestamp}"
            run_output_dir = os.path.join(config['folders']['output'], run_folder_name)
            if not os.path.exists(run_output_dir):
                os.makedirs(run_output_dir)
                
            update_console(f"Output folder created: {run_folder_name}")
            
            analyzer = VideoAnalyzer(config, st.session_state.ollama_svc)
            director = Director(config, st.session_state.ollama_svc)
            # Pass the specific output dir to processor
            processor = VideoProcessor(config, output_override=run_output_dir)
            
            # 2. Run Analysis
            update_console("Starting Analysis Pipeline...")
            progress_bar.progress(10)
            
            analysis_log = analyzer.run_full_analysis(temp_path, status_callback=update_console, context_guidance=user_guidance)
            progress_bar.progress(50)
            
            # Save Analysis Log
            log_path = os.path.join(config['folders']['logs'], f"{uploaded_file.name}_analysis.json")
            with open(log_path, "w", encoding='utf-8') as f:
                json.dump(analysis_log, f, indent=2, ensure_ascii=False)
            update_console(f"Analysis saved to {log_path}")
            
            # 3. Director Selection
            update_console("Director is selecting clips...")
            clips, rationale = director.select_viral_clips(analysis_log, status_callback=update_console, context_guidance=user_guidance)
            
            # Save Director Log
            director_log_path = os.path.join(config['folders']['logs'], f"{uploaded_file.name}_director.md")
            with open(director_log_path, "w", encoding='utf-8') as f:
                f.write(f"# Director's Rationale\n\n{rationale}")
            
            progress_bar.progress(70)
            
            if not clips:
                st.error("Director found no suitable clips! Check the logs.")
            else:
                st.success(f"Director selected {len(clips)} clips!")
                
                # 4. Render Clips
                cols = st.columns(len(clips))
                for i, clip in enumerate(clips):
                    status_text.text(f"Rendering Clip {i+1}/{len(clips)}...")
                    update_console(f"Rendering clip {i+1}: {clip['title']} ({clip['start_time']}-{clip['end_time']})")
                    
                    try:
                        out_path = processor.create_short(temp_path, clip, analysis_log['transcript'], i)
                        
                        with cols[i]:
                            st.subheader(f"Short #{i+1}")
                            st.caption(clip.get('title', 'Untitled'))
                            st.video(out_path)
                            st.write(f"**Score:** {clip.get('score')}/10")
                            st.write(f"_{clip.get('reason')}_")
                            
                            with open(out_path, "rb") as v_file:
                                st.download_button(
                                    label="Download MP4",
                                    data=v_file,
                                    file_name=os.path.basename(out_path),
                                    mime="video/mp4"
                                )
                    except Exception as e:
                        st.error(f"Failed to render clip {i+1}: {e}")
                        update_console(f"Error rendering clip {i+1}: {e}")

                progress_bar.progress(100)
                status_text.text("All Operations Complete.")
                st.balloons()
                
        except Exception as e:
            st.error(f"Critical Error: {str(e)}")
            update_console(f"CRITICAL FAIL: {str(e)}")

