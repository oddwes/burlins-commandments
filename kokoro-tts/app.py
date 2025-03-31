import os
import sys
import torch
import gradio as gr
from kokoro import KPipeline
import soundfile as sf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the pipeline
try:
    print("Loading Kokoro TTS model...")
    pipeline = KPipeline(lang_code='en')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def generate_speech(text):
    """
    Generate speech from text using the Kokoro TTS model.
    
    Args:
        text (str): Input text to convert to speech
        
    Returns:
        tuple: (sample_rate, audio_data)
    """
    try:
        # Generate audio using the pipeline
        generator = pipeline(text, voice='af_heart', speed=1)
        _, _, audio = next(generator)  # Get the first audio segment
        return 24000, audio  # Kokoro uses 24kHz sample rate
    except Exception as e:
        print(f"Error generating speech: {e}")
        raise gr.Error(f"Failed to generate speech: {str(e)}")

# Create Gradio interface
demo = gr.Interface(
    fn=generate_speech,
    inputs=gr.Textbox(placeholder="Enter text to convert to speech...", label="Input Text"),
    outputs=gr.Audio(label="Generated Speech"),
    title="Kokoro TTS",
    description="Convert text to speech using the Kokoro TTS model",
    examples=[
        "Hello, this is a text to speech demonstration.",
        "The quick brown fox jumps over the lazy dog.",
        "I hope you're having a wonderful day!"
    ]
)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)