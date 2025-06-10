from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os
from pathlib import Path

app = FastAPI(title="Disaster Classification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model_before = None
model_tuned = None
class_names = ['Earthquake', 'Land_Slide', 'Urban_Fire', 'Water_Disaster']

# Get the current directory (backend folder)
BACKEND_DIR = Path(__file__).parent
MODEL_BEFORE_PATH = BACKEND_DIR / "models" / "effnet_before.keras"
MODEL_TUNED_PATH = BACKEND_DIR / "models" / "effnet_tune.keras"

def load_models():
    global model_before, model_tuned
    try:
        print("Loading models...")
        print(f"Looking for models at:")
        print(f"  Before: {MODEL_BEFORE_PATH}")
        print(f"  Tuned: {MODEL_TUNED_PATH}")
        
        if not MODEL_BEFORE_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_BEFORE_PATH}")
        if not MODEL_TUNED_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_TUNED_PATH}")
            
        model_before = load_model(str(MODEL_BEFORE_PATH))
        model_tuned = load_model(str(MODEL_TUNED_PATH))
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise e

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize image to match model input size
    image = image.resize((260, 260))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to array and normalize
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply EfficientNet preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    return img_array

def get_prediction(model, img_array: np.ndarray) -> dict:
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = class_names[predicted_class_idx]
    
    # Get all probabilities
    all_predictions = {
        class_names[i]: float(predictions[0][i]) 
        for i in range(len(class_names))
    }
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_predictions": all_predictions
    }

@app.on_event("startup")
async def startup_event():
    """Load models when the app starts"""
    load_models()

@app.get("/")
async def root():
    return {"message": "Disaster Classification API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": model_before is not None and model_tuned is not None,
        "backend_dir": str(BACKEND_DIR),
        "model_paths": {
            "before": str(MODEL_BEFORE_PATH),
            "tuned": str(MODEL_TUNED_PATH)
        }
    }

@app.post("/predict")
async def predict_disaster(
    file: UploadFile = File(...),
    model_type: str = "tuned"
):
    """
    Predict disaster type from uploaded image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image
        img_array = preprocess_image(image)
        
        # Select model
        selected_model = model_tuned if model_type == "tuned" else model_before
        if selected_model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Get prediction
        result = get_prediction(selected_model, img_array)
        
        return JSONResponse(content={
            "success": True,
            "model_used": model_type,
            "filename": file.filename,
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "all_predictions": result["all_predictions"]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/compare")
async def compare_models(file: UploadFile = File(...)):
    """
    Compare predictions from both models
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image
        img_array = preprocess_image(image)
        
        # Get predictions from both models
        result_before = get_prediction(model_before, img_array)
        result_tuned = get_prediction(model_tuned, img_array)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "before_tuning": {
                "predicted_class": result_before["predicted_class"],
                "confidence": result_before["confidence"],
                "all_predictions": result_before["all_predictions"]
            },
            "after_tuning": {
                "predicted_class": result_tuned["predicted_class"],
                "confidence": result_tuned["confidence"],
                "all_predictions": result_tuned["all_predictions"]
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)