import os
import shutil
from pathlib import Path
import zipfile
from typing import List

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from src.triage.inference import ForensicScanner

app = FastAPI(title="ForenSURE-Net Web Dashboard")

# Ensure static and temp dirs exist
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# We will initialize the scanner globally (lazy-loaded or on start)
# For this demo, we assume the weights are at experiments/checkpoints/residual_stegnet_best.pth
WEIGHTS_PATH = Path("experiments/checkpoints/residual_stegnet_best.pth")

scanner = None

@app.on_event("startup")
def startup_event():
    global scanner
    # Default to single model for now. To use ensemble, add more models to this dict.
    models_dict = {}
    if WEIGHTS_PATH.exists():
        models_dict["LSB (SRNet)"] = {"weights": str(WEIGHTS_PATH)}
    
    if models_dict:
        scanner = ForensicScanner(models_dict)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(STATIC_DIR / "index.html", "r") as f:
        return f.read()

@app.post("/api/scan")
async def scan_files(files: List[UploadFile] = File(...)):
    if not scanner:
        return {"error": "Scanner not initialized. Model weights not found."}

    # Create a unique job directory
    import uuid
    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir()
    
    try:
        # Save uploaded files
        for file in files:
            file_path = job_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # If ZIP, extract it
            if file.filename.endswith(".zip"):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(job_dir)
                os.remove(file_path) # Remove zip after extraction

        # Run Scan
        results = scanner.scan_directory(job_dir)
        
        # Cleanup
        shutil.rmtree(job_dir, ignore_errors=True)
        
        return {"status": "success", "results": results}
        
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
