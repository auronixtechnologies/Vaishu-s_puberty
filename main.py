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

# Serve images locally (fallback when Drive map is missing)
if os.path.exists(PHOTOS_DIR):
    app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")

# Load Google Drive CDN links
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
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchResponse(BaseModel):
    matches: list[dict]
    message: str

# Load OpenFace DNN once on startup
OPENFACE_MODEL = os.path.join(os.path.dirname(__file__), "openface.t7")
if not os.path.exists(OPENFACE_MODEL):
    raise RuntimeError(f"openface.t7 not found at {OPENFACE_MODEL}")

openface_net = cv2.dnn.readNetFromTorch(OPENFACE_MODEL)
openface_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
openface_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_128d_embedding(img_bgr):
    """Return L2-normalised 128-D OpenFace vector for the largest face, or None."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]
    pad   = int(0.1 * max(w, h))
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, x-pad);  y1 = max(0, y-pad)
    x2 = min(w_img, x+w+pad); y2 = min(h_img, y+h+pad)
    face  = img_bgr[y1:y2, x1:x2]
    blob  = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0,0,0), swapRB=True, crop=False)
    openface_net.setInput(blob)
    vec   = openface_net.forward()[0]
    return vec / (np.linalg.norm(vec) + 1e-8)

# ── Pre-load the entire embedding database once on startup  ──────────────────
_representations: list = []

def _load_representations():
    global _representations
    pkl_files = glob.glob(os.path.join(PHOTOS_DIR, "*.pkl"))
    if not pkl_files:
        print(f"[WARNING] No .pkl file in {PHOTOS_DIR} - /api/match will return empty results.")
        return
    with open(pkl_files[0], "rb") as f:
        _representations = pickle.load(f)
    print(f"[OK] Loaded {len(_representations)} face embeddings from {os.path.basename(pkl_files[0])}")

_load_representations()
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(PHOTOS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename,
                        media_type="application/octet-stream")


@app.post("/api/match", response_model=MatchResponse)
async def match_face(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        nparr   = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode the uploaded image.")

        target_vec = get_128d_embedding(img_bgr)
        if target_vec is None:
            return {"matches": [], "message": "No face detected. Please try a clear, front-facing selfie."}

        if not _representations:
            raise HTTPException(status_code=500, detail="Embedding database not loaded.")

        # Cosine distance; diagnosis shows genuine matches < 0.20, strangers cluster > 0.50
        # Threshold of 0.25 gives tight, accurate results
        THRESHOLD = 0.25

        scored = []
        for r in _representations:
            db_vec  = np.array(r["embedding"], dtype=np.float32)
            db_vec  = db_vec / (np.linalg.norm(db_vec) + 1e-8)
            dist    = float(1.0 - np.dot(target_vec, db_vec))
            if dist <= THRESHOLD:
                # Use replace+split so Windows paths (D:\...) work on Linux too
                raw_path = r["identity"]
                fname = raw_path.replace("\\", "/").split("/")[-1]
                scored.append({"filename": fname, "distance": dist})

        scored.sort(key=lambda x: x["distance"])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    # De-duplicate: keep only the best-distance entry per unique filename
    seen = set()
    final_matches = []
    for match in scored:
        fn = match["filename"]
        if fn in seen:
            continue
        seen.add(fn)

        if drive_map and fn in drive_map:
            drive_id = drive_map[fn]
            image_url    = f"https://lh3.googleusercontent.com/d/{drive_id}=w1000"
            download_url = f"https://drive.google.com/uc?export=download&id={drive_id}"
        else:
            image_url    = f"{BASE_URL}/photos/{fn}"
            download_url = f"{BASE_URL}/api/download/{fn}"

        final_matches.append({
            "url":          image_url,
            "download_url": download_url,
            "filename":     fn,
            "confidence":   round((1.0 - match["distance"]) * 100, 1),
        })

    msg = (
        f"Found {len(final_matches)} matching photos! Sorted by AI confidence."
        if final_matches
        else "No matching photos found. Try a clearer, front-facing selfie."
    )
    return {"matches": final_matches, "message": msg}


@app.get("/api/videos")
async def get_videos():
    return {"videos": []}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 10000)))