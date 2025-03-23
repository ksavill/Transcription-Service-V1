import traceback
import torch
from typing import Any

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

def load_model(model_name: str) -> Any | None:
    try:
        import whisper
        # Dynamically choose the device based on CUDA availability.
        device = "cuda" if cuda_available else "cpu"
        print(f"Loading model '{model_name}' on device: {device}")
        local_model = whisper.load_model(model_name, device=device)
        return local_model
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Failed to load model: {e}\n\n{tb}")
        return None

if __name__ == '__main__':
    model_name = "base"
    model = load_model(model_name)
    print(f"{'Model loaded successfully' if model else 'Failed to load model'}")
