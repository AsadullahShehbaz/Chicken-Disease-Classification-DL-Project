import os
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse  # to show a simple upload page
import shutil                               # to save uploaded file temporarily

from src.cnnClassifier.pipeline.predict import PredictionPipeline

# Create FastAPI app
app = FastAPI(title="Chicken Disease Classifier")


# ── Route 1: Home page with simple HTML upload form ──────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Chicken Disease Classifier</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input type="file" name="file">
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """


# ── Route 2: Prediction endpoint ─────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Step 1: Save uploaded image temporarily to disk
    temp_path = os.path.join("temp_uploads", file.filename)
    os.makedirs("temp_uploads", exist_ok=True)  # create folder if not exists

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)   # write uploaded bytes to file

    # Step 2: Pass saved image path to your existing PredictionPipeline
    pipeline = PredictionPipeline(filename=temp_path)
    result = pipeline.predict()                 # returns [{"image": "Healthy"}]

    # Step 3: Cleanup temp file after prediction
    os.remove(temp_path)

    # Step 4: Return the result
    return result


# ── Run server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)


## Folder Structure Should Look Like This
"""
project/
├── app.py                          ← this file
├── artifacts/
│   └── training/
│       └── model.keras
└── src/
    └── cnnClassifier/
        └── pipeline/
            └── predict.py          ← your existing file
"""