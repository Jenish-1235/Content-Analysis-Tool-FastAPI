from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import socket
from pathlib import Path
import os
from typing import Optional
from safetensors.torch import load_file

# Replace this with your actual model class definition
class YourModelClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Example model architecture; replace with actual layers
        self.layer1 = torch.nn.Linear(256, 128)
        self.layer2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

app = FastAPI(
    title="Gender Bias Detection API",
    description="API for detecting gender bias in text content",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

# Model setup
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.safetensors")

def load_model():
    try:
        global model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        # Load model state from safetensors file
        state_dict = load_file(MODEL_PATH)
        
        # Initialize your model class
        model = YourModelClass()
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Pydantic models
class TextRequest(BaseModel):
    text: str

class BiasResponse(BaseModel):
    has_bias: bool
    bias_score: float
    bias_type: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    if not load_model():
        print("Warning: Model failed to load")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/detect-bias")
async def detect_bias(request: TextRequest):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not initialized. Please ensure model file exists and is valid."
        )
    try:
        # Convert text input to a fixed-size tensor
        input_text = [ord(char) for char in request.text]
        input_text = input_text[:256] + [0] * max(0, 256 - len(input_text))  # Pad or truncate
        input_tensor = torch.tensor(input_text).float().unsqueeze(0)
        
        with torch.no_grad():
            bias_score = model(input_tensor).item()
        
        threshold = 0.5  # Adjust this threshold based on your model's requirements
        
        return BiasResponse(
            has_bias=bias_score > threshold,
            bias_score=float(bias_score),
            bias_type="gender_bias" if bias_score > threshold else None
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text: {str(e)}"
        )


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if __name__ == "__main__":
    import uvicorn
    port = 8000
    while is_port_in_use(port):
        port += 1
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="127.0.0.1", port=port)