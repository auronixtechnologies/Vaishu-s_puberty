from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from deepface import DeepFace
import numpy as np
import cv2
import os
import json

app = FastAPI(title="Face Recognition Gallery API")

# We use a local folder name that will exist on the Render server
PHOTOS_DIR = "./photos_db"

# Serve local photos so they can be viewed in the browser during local testing ONLY if Google Drive links aren't used
if os.path.exists(PHOTOS_DIR):
    app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")

# Cloud Deployment: Load Google Drive mappings
DRIVE_MAP_FILE = "drive_links.json"
drive_map = {}
if os.path.exists(DRIVE_MAP_FILE):
    with open(DRIVE_MAP_FILE, "r", encoding="utf-8") as f:
        drive_map = json.load(f)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vaishu-puberty.netlify.app",
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchResponse(BaseModel):
    matches: list[dict]
    message: str

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(PHOTOS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    # Setting media_type to application/octet-stream tricks the browser into strictly downloading the file
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

@app.post("/api/match", response_model=MatchResponse)
async def match_face(file: UploadFile = File(...)):
    contents = await file.read()
    
    try:
        # Load image into BGR format for OpenCV/DeepFace
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # In True Cloud mode, we only generate the mathematical extraction of the uploaded selfie
        face_objs = DeepFace.represent(img_path=img_bgr, model_name="ArcFace", enforce_detection=False)
        
        if len(face_objs) == 0:
            return {"matches": [], "message": "No closely matching faces found."}
            
        target_embedding = np.array(face_objs[0]["embedding"])
        
        # Load the pre-calculated math models directly, bypassing DeepFace folder verification
        import glob
        pkl_files = glob.glob(os.path.join(PHOTOS_DIR, "*.pkl"))
        if not pkl_files:
            raise HTTPException(status_code=500, detail=f"AI Dictionary (.pkl file) missing from {PHOTOS_DIR}!!")
            
        import pickle
        with open(pkl_files[0], "rb") as f:
            representations = pickle.load(f)
            
        scored_matches = []
        for rep in representations:
            # Modern DeepFace stores representations as dictionaries in a list
            file_path = rep["identity"]
            db_emb = np.array(rep["embedding"])
            
            # Pure Cosine Similarity Distance (closer to 0 is an exact match)
            distance = np.dot(target_embedding, db_emb) / (np.linalg.norm(target_embedding) * np.linalg.norm(db_emb))
            # Convert cosine similarity to distance (0 = identical)
            distance = 1.0 - distance 
            
            filename = os.path.basename(file_path)
            scored_matches.append({
                "filename": filename,
                "distance": distance
            })
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    # Sort matches by pure identicalness (Threshold for ArcFace is usually under 0.68)
    scored_matches.sort(key=lambda x: x["distance"])
    
    seen = set()
    final_matches = []
    # Filter only those that are a legitimate mathematical match
    for match in scored_matches:
        if match["distance"] > 0.68:
            continue
            
        filename = match["filename"]
        if filename not in seen:
            seen.add(filename)
            
            # Switch to Google Drive architecture if we have the drive links map (Cloud Mode)
            if drive_map and filename in drive_map:
                drive_id = drive_map[filename]
                # Google Drive direct view/download streams
                # Using lh3.googleusercontent.com CDN totally bypasses strict browser image redirects!
                image_url = f"https://lh3.googleusercontent.com/d/{drive_id}=w1000"
                download_url = f"https://drive.google.com/uc?export=download&id={drive_id}"
            else:
                # Fallback to local mode testing
                image_url = f"http://localhost:8000/photos/{filename}"
                download_url = f"http://localhost:8000/api/download/{filename}"
            
            final_matches.append({
                "url": image_url,
                "download_url": download_url,
                "filename": filename
            })
            
    return {"matches": final_matches, "message": f"Found {len(final_matches)} photos! Sorted by AI confidence."}

@app.get("/api/videos")
async def get_videos():
    return {"videos": []}

import os

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
