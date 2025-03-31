from huggingface_hub import snapshot_download
import os
import argparse
from dotenv import load_dotenv

def download_model_weights(token=None):
    """
    Download model weights from Hugging Face Hub.
    
    Args:
        token (str, optional): Hugging Face token for authentication
    """
    # Correct repository name
    MODEL_ID = "hexgrad/Kokoro-82M"
    
    print(f"Downloading Kokoro TTS model weights from {MODEL_ID}...")
    try:
        snapshot_download(
            repo_id=MODEL_ID, 
            local_dir="./weights",
            token=token
        )
        print("Model weights downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model weights: {e}")
        print("\nPlease make sure your Hugging Face token is correct.")
        print("You can get a token from: https://huggingface.co/settings/tokens")

if __name__ == "__main__":
    # Load token from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Download Kokoro TTS model weights")
    parser.add_argument("--token", type=str, help="Hugging Face token for authentication")
    args = parser.parse_args()
    
    # First try to use token from command line args
    # Then try to use token from .env file
    # Then try to use token from environment variable
    token = args.token or os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    download_model_weights(token)