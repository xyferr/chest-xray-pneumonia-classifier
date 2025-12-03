"""
FastAPI Inference Service for Chest X-Ray Pneumonia Classification.

Endpoints:
- GET  /health     - Health check
- GET  /info       - Model information
- POST /predict    - Predict pneumonia from chest X-ray image
- POST /predict/gradcam - Predict with Grad-CAM visualization

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
import sys
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

# Add src to path for imports
SRC_PATH = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from models import create_model, SUPPORTED_MODELS
from dataset import get_inference_transforms, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES
from utils import get_device, get_logger, InferenceConfig


# =============================================================================
# Configuration
# =============================================================================

# Model configuration - loaded from .env file or environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/checkpoints/best_model.pt")
MODEL_NAME = os.environ.get("MODEL_NAME", "baseline_cnn")
DEVICE = os.environ.get("DEVICE", "auto")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))

# Initialize logger
logger = get_logger("inference")


# =============================================================================
# Response Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="healthy")
    timestamp: str = Field(..., example="2025-11-30T12:00:00")
    model_loaded: bool = Field(..., example=True)


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = Field(..., example="resnet50")
    model_path: str = Field(..., example="outputs/checkpoints/best_model.pt")
    device: str = Field(..., example="cuda")
    image_size: int = Field(..., example=224)
    classes: List[str] = Field(..., example=["NORMAL", "PNEUMONIA"])
    supported_formats: List[str] = Field(..., example=["jpg", "jpeg", "png", "bmp"])


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: str = Field(..., example="PNEUMONIA")
    confidence: float = Field(..., ge=0, le=1, example=0.95)
    probabilities: Dict[str, float] = Field(
        ..., 
        example={"NORMAL": 0.05, "PNEUMONIA": 0.95}
    )
    processing_time_ms: float = Field(..., example=45.2)


class GradCAMResponse(PredictionResponse):
    """Prediction response with Grad-CAM visualization."""
    gradcam_image: str = Field(
        ..., 
        description="Base64 encoded Grad-CAM overlay image"
    )


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., example="Invalid image format")
    detail: Optional[str] = Field(None, example="Expected JPEG or PNG image")


# =============================================================================
# Model Loading
# =============================================================================

class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.transform = None
        self.gradcam = None
        self.config = None
        self.loaded = False
    
    def load_model(
        self,
        model_path: str = MODEL_PATH,
        model_name: str = MODEL_NAME,
        device_str: str = DEVICE,
    ) -> bool:
        """
        Load the model for inference.
        
        Args:
            model_path: Path to model checkpoint
            model_name: Model architecture name
            device_str: Device to use ('auto', 'cuda', 'cpu', 'mps')
        
        Returns:
            True if model loaded successfully
        """
        try:
            # Setup device
            if device_str == "auto":
                self.device = get_device()
            else:
                self.device = torch.device(device_str)
            
            logger.info(f"Loading model {model_name} from {model_path}")
            logger.info(f"Using device: {self.device}")
            
            # Create model
            self.model = create_model(
                model_name=model_name,
                num_classes=2,
                pretrained=False,
            )
            
            # Load weights
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                logger.info(f"Loaded weights from {model_path}")
            else:
                logger.warning(f"Model path not found: {model_path}")
                logger.warning("Using randomly initialized weights (for testing only)")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self.transform = get_inference_transforms(IMAGE_SIZE)
            
            # Store config
            self.config = InferenceConfig(
                model_path=model_path,
                model_name=model_name,
                image_size=IMAGE_SIZE,
            )
            
            # Setup Grad-CAM
            self._setup_gradcam()
            
            self.loaded = True
            logger.info("Model loaded successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.loaded = False
            return False
    
    def _setup_gradcam(self):
        """Setup Grad-CAM for visualization."""
        try:
            from eval import GradCAM, get_gradcam_target_layer
            
            target_layer = get_gradcam_target_layer(self.model, self.config.model_name)
            self.gradcam = GradCAM(self.model, target_layer)
            logger.info("Grad-CAM initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Grad-CAM: {e}")
            self.gradcam = None
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: PIL Image
        
        Returns:
            Preprocessed tensor
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Make prediction on an image.
        
        Args:
            image: PIL Image
        
        Returns:
            Dictionary with prediction results
        """
        import time
        start_time = time.time()
        
        # Preprocess
        tensor = self.preprocess_image(image)
        
        # Inference
        output = self.model(tensor)
        probs = F.softmax(output, dim=1)
        
        # Get prediction
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
        
        # Processing time
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "probabilities": {
                CLASS_NAMES[i]: probs[0, i].item() 
                for i in range(len(CLASS_NAMES))
            },
            "processing_time_ms": processing_time,
        }
    
    def predict_with_gradcam(self, image: Image.Image) -> Dict[str, Any]:
        """
        Make prediction with Grad-CAM visualization.
        
        Args:
            image: PIL Image
        
        Returns:
            Dictionary with prediction and Grad-CAM image
        """
        import time
        start_time = time.time()
        
        if self.gradcam is None:
            raise ValueError("Grad-CAM not initialized")
        
        # Preprocess - create a fresh copy for each prediction
        tensor = self.preprocess_image(image)
        
        # Get Grad-CAM - this now returns probabilities too
        cam, pred_idx, confidence, probabilities = self.gradcam(tensor)
        
        # Create overlay image
        gradcam_image = self._create_gradcam_overlay(image, cam)
        
        # Encode to base64
        buffered = io.BytesIO()
        gradcam_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Processing time
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "probabilities": probabilities,
            "processing_time_ms": processing_time,
            "gradcam_image": img_base64,
        }
    
    def _create_gradcam_overlay(
        self, 
        original_image: Image.Image, 
        cam: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """
        Create Grad-CAM overlay on original image.
        
        Args:
            original_image: Original PIL Image
            cam: Grad-CAM heatmap
            alpha: Overlay transparency
        
        Returns:
            PIL Image with Grad-CAM overlay
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Resize CAM to image size
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                original_image.size, Image.BILINEAR
            )
        ) / 255.0
        
        # Apply colormap
        heatmap = cm.jet(cam_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = Image.fromarray(heatmap)
        
        # Convert original to RGB
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # Blend images
        overlay = Image.blend(original_image, heatmap, alpha)
        
        return overlay


# Global model manager
model_manager = ModelManager()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Chest X-Ray Pneumonia Classifier API",
    description="""
    An AI-powered API for detecting pneumonia from chest X-ray images.
    
    ## Features
    - **Fast inference** using PyTorch with GPU support
    - **Grad-CAM visualizations** for model interpretability
    - **REST API** for easy integration
    
    ## Usage
    1. Upload a chest X-ray image to `/predict`
    2. Get diagnosis (NORMAL/PNEUMONIA) with confidence score
    3. Optionally get Grad-CAM heatmap showing model focus areas
    
    ## Model
    Trained on the Kaggle Chest X-Ray Pneumonia dataset with ResNet-50 backbone.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting up API server...")
    
    success = model_manager.load_model(
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
        device_str=DEVICE,
    )
    
    if not success:
        logger.warning("Model loading failed - API will return errors for predictions")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with simple HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chest X-Ray Pneumonia Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 { color: #333; }
            .container {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .upload-form {
                margin: 20px 0;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #45a049;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                display: none;
            }
            .normal { background-color: #d4edda; color: #155724; }
            .pneumonia { background-color: #f8d7da; color: #721c24; }
            #preview, #gradcam {
                max-width: 300px;
                margin: 10px;
                border-radius: 5px;
            }
            .images {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
            }
            .loader {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                display: none;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Chest X-Ray Pneumonia Classifier</h1>
            <p>Upload a chest X-ray image to detect pneumonia using AI.</p>
            
            <div class="upload-form">
                <input type="file" id="imageInput" accept="image/*">
                <br>
                <label>
                    <input type="checkbox" id="gradcamCheckbox"> Include Grad-CAM visualization
                </label>
                <br><br>
                <button onclick="predict()">🔍 Analyze Image</button>
                <div class="loader" id="loader"></div>
            </div>
            
            <div class="images">
                <div>
                    <h3>Uploaded Image</h3>
                    <img id="preview" style="display:none;">
                </div>
                <div id="gradcamContainer" style="display:none;">
                    <h3>Grad-CAM Heatmap</h3>
                    <img id="gradcam">
                </div>
            </div>
            
            <div id="result"></div>
            
            <hr>
            <p>
                <a href="/docs">📚 API Documentation</a> | 
                <a href="/health">❤️ Health Check</a> |
                <a href="/info">ℹ️ Model Info</a>
            </p>
        </div>
        
        <script>
            document.getElementById('imageInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('preview').src = e.target.result;
                        document.getElementById('preview').style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            });
            
            async function predict() {
                const fileInput = document.getElementById('imageInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image first!');
                    return;
                }
                
                const useGradcam = document.getElementById('gradcamCheckbox').checked;
                const endpoint = useGradcam ? '/predict/gradcam' : '/predict';
                
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('loader').style.display = 'inline-block';
                document.getElementById('result').style.display = 'none';
                document.getElementById('gradcamContainer').style.display = 'none';
                
                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        const resultDiv = document.getElementById('result');
                        const className = data.prediction.toLowerCase();
                        resultDiv.className = className;
                        resultDiv.innerHTML = `
                            <h3>Diagnosis: ${data.prediction}</h3>
                            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Normal probability:</strong> ${(data.probabilities.NORMAL * 100).toFixed(1)}%</p>
                            <p><strong>Pneumonia probability:</strong> ${(data.probabilities.PNEUMONIA * 100).toFixed(1)}%</p>
                            <p><em>Processing time: ${data.processing_time_ms.toFixed(1)}ms</em></p>
                        `;
                        resultDiv.style.display = 'block';
                        
                        if (data.gradcam_image) {
                            document.getElementById('gradcam').src = 'data:image/png;base64,' + data.gradcam_image;
                            document.getElementById('gradcamContainer').style.display = 'block';
                        }
                    } else {
                        alert('Error: ' + (data.detail || data.error || 'Unknown error'));
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.getElementById('loader').style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_manager.loaded else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model_manager.loaded,
    )


@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        device=str(model_manager.device) if model_manager.device else "not loaded",
        image_size=IMAGE_SIZE,
        classes=CLASS_NAMES,
        supported_formats=["jpg", "jpeg", "png", "bmp", "gif"],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image file"),
):
    """
    Predict pneumonia from a chest X-ray image.
    
    - **file**: Chest X-ray image (JPEG, PNG, BMP, GIF)
    
    Returns prediction (NORMAL/PNEUMONIA) with confidence score.
    """
    # Check if model is loaded
    if not model_manager.loaded:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check server logs.",
        )
    
    # Validate file type
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "image/bmp", "image/gif"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Expected image file.",
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        result = model_manager.predict(image)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/predict/gradcam",
    response_model=GradCAMResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def predict_with_gradcam(
    file: UploadFile = File(..., description="Chest X-ray image file"),
):
    """
    Predict pneumonia with Grad-CAM visualization.
    
    - **file**: Chest X-ray image (JPEG, PNG, BMP, GIF)
    
    Returns prediction with Grad-CAM heatmap showing model focus areas.
    The heatmap helps explain what regions the model is looking at.
    """
    # Check if model is loaded
    if not model_manager.loaded:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check server logs.",
        )
    
    # Check if Grad-CAM is available
    if model_manager.gradcam is None:
        raise HTTPException(
            status_code=500,
            detail="Grad-CAM not available for this model.",
        )
    
    # Validate file type
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "image/bmp", "image/gif"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Expected image file.",
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction with Grad-CAM
        result = model_manager.predict_with_gradcam(image)
        
        return GradCAMResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
