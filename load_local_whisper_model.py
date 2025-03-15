import traceback

def load_model(model_name: str) -> bool:
    try:
        import whisper
        whisper.load_model(model_name)
        return True
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Failed to load model: {e}\n\n {tb}")
        return False
    
if __name__ == '__main__':
    model_name = "base"
    success = load_model(model_name)
    print(f"{'Model loaded successfully' if success else 'Failed to load model'}")