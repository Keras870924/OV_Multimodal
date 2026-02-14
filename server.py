"""
FastAPI inference server for the Ovarian Cancer Multimodal Classifier.

Exposes REST API endpoints for health checking and ensemble prediction.
Designed to be launched by the Electron GUI or run standalone.

Usage:
    python server.py
    # or
    uvicorn server:app --host 127.0.0.1 --port 8000
"""

import base64
import io
import os
import sys
import tempfile
import traceback

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Initialize the FastAPI application
app = FastAPI(
    title="Ovarian Cancer Multimodal Classifier API",
    description="Ensemble prediction using Swin-T + DenseNet121 + Bio_ClinicalBERT",
    version="1.0.0",
)

# Allow CORS for the Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine (loaded on startup)
inference_engine = None


@app.on_event("startup")
async def load_models():
    """Load ensemble models and text processor on server startup."""
    global inference_engine
    from src.inference import MultimodalInference

    print("Initializing inference engine...")
    inference_engine = MultimodalInference(
        model_dir=os.environ.get("MODEL_DIR", "log"),
        metadata_path=os.environ.get("METADATA_PATH", "Clinical_data.csv"),
    )
    print("Inference engine ready.")


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        JSON with status and number of loaded models.
    """
    if inference_engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "message": "Models are still loading."},
        )
    return {
        "status": "ok",
        "models_loaded": len(inference_engine.models),
        "device": str(inference_engine.device),
    }


@app.post("/preview")
async def preview_image(
    image: UploadFile = File(..., description="Ultrasound image (DICOM/PNG/JPG)"),
):
    """Convert an uploaded image (including DICOM) to a base64-encoded PNG for preview.

    Returns:
        JSON with ``image`` key containing a data-URI string.
    """
    try:
        suffix = os.path.splitext(image.filename)[1] if image.filename else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await image.read()
            tmp.write(contents)
            tmp_path = tmp.name

        from PIL import Image as PILImage

        if suffix.lower() == ".dcm":
            from src.data.dicom_utils import read_dicom_file

            pil_img = read_dicom_file(tmp_path)
        else:
            pil_img = PILImage.open(tmp_path).convert("RGB")

        os.unlink(tmp_path)

        if pil_img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to read the image file."},
            )

        # Resize for preview (max 512px on the longest side)
        max_side = 512
        w, h = pil_img.size
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            pil_img = pil_img.resize(
                (int(w * ratio), int(h * ratio)), PILImage.LANCZOS
            )

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"image": f"data:image/png;base64,{b64}"}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Ultrasound image (DICOM/PNG/JPG)"),
    age: float = Form(..., description="Patient age in years"),
    transducer_type: str = Form(
        ..., description="Transducer type: Trandabdominal or Transvaginal"
    ),
    report_text: str = Form("", description="Ultrasound report text"),
):
    """Run ensemble prediction on an uploaded ultrasound image with clinical data.

    Args:
        image: Uploaded ultrasound image file (DICOM, PNG, or JPG).
        age: Patient age in years.
        transducer_type: Ultrasound transducer type.
        report_text: Free-text ultrasound report.

    Returns:
        JSON with ensemble prediction, per-class probabilities, and per-fold details.
    """
    if inference_engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Models are still loading. Please try again shortly."},
        )

    try:
        # Save uploaded file to a temporary location
        suffix = os.path.splitext(image.filename)[1] if image.filename else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await image.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Run ensemble prediction
        result = inference_engine.predict(
            image=tmp_path,
            age=age,
            transducer_type=transducer_type,
            report_text=report_text,
        )

        # Clean up temporary file
        os.unlink(tmp_path)

        return result

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
