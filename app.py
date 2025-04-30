from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from pathlib import Path

app = FastAPI(title="Chest X-Ray Classification System")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables
MODEL_PATH = "chest_xray_best.keras"  # Updated to match train_eval_tf.py
model = None
IMG_SIZE = (224, 224)  # Match the size used in training

# Decision threshold calibrated during training
BEST_THR_PATH = Path("best_threshold.npy")
BEST_THR = float(np.load(BEST_THR_PATH)) if BEST_THR_PATH.exists() else 0.5

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        # No need to compile for inference-only use
    return model

def preprocess_image(image: Image.Image):
    # Resize and normalize image as done in training
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img).astype(np.float32)  # model's Rescaling layer handles /255
    return img_array[None, ...]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Get prediction
        model = load_model()
        prediction = model.predict(processed_image)
        probability = float(prediction[0][0])
        
        is_pneumonia = probability >= BEST_THR
        result = {
            "probability": probability,
            "threshold": BEST_THR,
            "prediction": "PNEUMONIA" if is_pneumonia else "NORMAL",
            "confidence": f"{(probability if is_pneumonia else 1 - probability)*100:.2f}%"
        }
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/model-performance")
async def get_model_performance():
    try:
        model = load_model()
        
        # Get model summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        
        # Return metrics that match what's tracked in training
        metrics = {
            "accuracy": 0.95,  # Placeholder values - will be updated from actual model performance
            "auc": 0.97,       # These should be updated based on your model's actual performance
        }
        
        return {
            "model_summary": "\n".join(model_summary),
            "metrics": metrics
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)  # Match port in run.sh 