import ollama
import time
import logging

class OllamaService:
    def __init__(self):
        self.current_model = None

    def list_local_models(self):
        """Returns complex objects with metadata to identify vision models."""
        try:
            models = ollama.list()
            # Handle potential API response variations
            return models.get('models', [])
        except Exception as e:
            logging.error(f"Error fetching models: {e}")
            return []

    def get_vision_models(self):
        """
        Filters local models to find those likely to support vision.
        Checks for 'clip' family or specific naming conventions.
        """
        all_models = self.list_local_models()
        vision_models = []
        
        for m in all_models:
            name = m.model
            
            # 1. Easy check: Name contains clues
            if any(x in name.lower() for x in ['llava', 'vision', 'vl', 'minicpm', 'bakllava']):
                vision_models.append(name)
                continue
                
            # 2. Hard check: Inspect details (slower, but accurate for custom models)
            try:
                details = ollama.show(name)
                # Check for 'clip' in families (common for LLaVA based)
                # details is a ShowResponse object, details.details is ModelDetails object
                families = details.details.families if details.details and details.details.families else []
                if families and 'clip' in families:
                    vision_models.append(name)
                    continue
                    
                # Check if model info/template mentions images
                if 'images' in str(details.modelfile):
                    vision_models.append(name)
            except:
                pass
                
        return vision_models if vision_models else [m.model for m in all_models]

    def load_model(self, model_name):
        """
        Explicitly loads a model. 
        In actual Ollama API, sending a request triggers load.
        We can force a small keep_alive or just trigger a dry-run to load.
        """
        if self.current_model == model_name:
            return
        
        print(f"🔄 Switching Model: {self.current_model} -> {model_name}")
        # Explicit unload is difficult in pure Ollama API without 
        # setting keep_alive to 0 on the previous model.
        if self.current_model:
            self.unload_model(self.current_model)
            
        self.current_model = model_name
        # Pre-warm
        try:
            ollama.chat(model=model_name, messages=[])
        except:
            pass 

    def unload_model(self, model_name):
        """
        Forces a model to unload by sending a request with keep_alive=0
        This is CRITICAL for VRAM management.
        """
        try:
            print(f"🔻 Unloading {model_name} from VRAM...")
            # Sending an empty request with keep_alive=0 unloads it immediately
            ollama.chat(model=model_name, keep_alive=0)
            self.current_model = None
        except Exception as e:
            print(f"Warning: Could not unload model {model_name}: {e}")

    def analyze_image(self, model, image_path, prompt="Describe this image"):
        self.load_model(model)
        try:
            # Enforce GPU usage via options
            # num_gpu: 999 instructs Ollama to offload ALL layers to GPU.
            # If the system cannot support this, behavior depends on Ollama server config,
            # but this is the strongest request we can make from the client.
            options = {
                "num_gpu": 999,
                "temperature": 0.2 
            }
            
            res = ollama.chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }],
                options=options
            )
            return res.message.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def generate_text(self, model, prompt):
        self.load_model(model)
        try:
            options = {
                "num_gpu": 999,
                "temperature": 0.7 
            }
            
            res = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options=options
            )
            return res.message.content
        except Exception as e:
            return f"Error generating text: {str(e)}"
