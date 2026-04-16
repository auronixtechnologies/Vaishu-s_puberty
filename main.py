from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import cv2
import os
import json
import glob
import pickle

app = FastAPI(title="Face Recognition Gallery API")

PHOTOS_DIR = "./photos_db"
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Serve local photos
if os.path.exists(PHOTOS_DIR):
    app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")

# Load drive map
DRIVE_MAP_FILE = "drive_links.json"
drive_map = {}
if os.path.exists(DRIVE_MAP_FILE):
    with open(DRIVE_MAP_FILE, "r", encoding="utf-8") as f:
        drive_map = json.load(f)

# CORS
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
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')


# 🔥 Lightweight embedding
def get_embedding(img_bgr):
    img = cv2.resize(img_bgr, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    embedding = img.flatten().astype("float32")
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@app.post("/api/match", response_model=MatchResponse)
async def match_face(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        target_embedding = get_embedding(img_bgr)

        pkl_files = glob.glob(os.path.join(PHOTOS_DIR, "*.pkl"))
        if not pkl_files:
            raise HTTPException(status_code=500, detail="No embedding file found")

        with open(pkl_files[0], "rb") as f:
            representations = pickle.load(f)

        scored_matches = []

        for rep in representations:
            db_emb = np.array(rep["embedding"])
            db_emb = db_emb[:len(target_embedding)]

            similarity = np.dot(target_embedding, db_emb) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(db_emb)
            )

            distance = 1 - similarity
            filename = os.path.basename(rep["identity"])

            scored_matches.append({
                "filename": filename,
                "distance": float(distance)
            })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    scored_matches.sort(key=lambda x: x["distance"])

    final_matches = []
    for match in scored_matches[:5]:
        filename = match["filename"]

        if drive_map and filename in drive_map:
            drive_id = drive_map[filename]
            image_url = f"https://lh3.googleusercontent.com/d/{drive_id}=w1000"
            download_url = f"https://drive.google.com/uc?export=download&id={drive_id}"
        else:
            image_url = f"{BASE_URL}/photos/{filename}"
            download_url = f"{BASE_URL}/api/download/{filename}"

        final_matches.append({
            "url": image_url,
            "download_url": download_url,
            "filename": filename
        })

    return {"matches": final_matches, "message": "Lightweight match complete"}


@app.get("/api/videos")
async def get_videos():
    return {"videos": []}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
