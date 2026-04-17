"""
index_photos.py  -  Build the face embedding database using OpenFace DNN (128-D).
Run this locally whenever you add new photos, then upload the updated .pkl to your server.

Usage:
    python index_photos.py
"""

import os
import cv2
import numpy as np
import pickle
import glob
import time

PHOTOS_DIR = r"D:\Vaishu\Photos"
OUTPUT_DIR = r"D:\Vaishu\app\backend\photos_db"
PKL_FILE   = os.path.join(OUTPUT_DIR, "face_embeddings.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading OpenFace DNN model...")
OPENFACE_MODEL = os.path.join(os.path.dirname(__file__), "openface.t7")
net = cv2.dnn.readNetFromTorch(OPENFACE_MODEL)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

image_files = (
    glob.glob(os.path.join(PHOTOS_DIR, "*.jpg"))  +
    glob.glob(os.path.join(PHOTOS_DIR, "*.JPG"))  +
    glob.glob(os.path.join(PHOTOS_DIR, "*.jpeg")) +
    glob.glob(os.path.join(PHOTOS_DIR, "*.png"))  +
    glob.glob(os.path.join(PHOTOS_DIR, "*.PNG"))
)

total = len(image_files)
print(f"Found {total} photos - starting indexing...\n")

embeddings = []
start = time.time()

for idx, img_path in enumerate(image_files):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue

    h_img, w_img = img_bgr.shape[:2]
    # Resize large images for faster processing
    scale = 800 / max(h_img, w_img)
    if scale < 1:
        img_small = cv2.resize(img_bgr, (int(w_img*scale), int(h_img*scale)))
    else:
        img_small = img_bgr
        scale = 1.0

    gray  = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))

    for (x, y, w, h) in faces:
        # Scale coordinates back to original image size
        x1 = int(max(0, (x - 0.1*w) / scale))
        y1 = int(max(0, (y - 0.1*h) / scale))
        x2 = int(min(w_img, (x + w*1.1) / scale))
        y2 = int(min(h_img, (y + h*1.1) / scale))

        face_crop = img_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # OpenFace requires 96x96 RGB input
        blob = cv2.dnn.blobFromImage(face_crop, 1.0/255, (96, 96),
                                     (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        vec = net.forward()[0]   # 128-D vector
        vec = vec / (np.linalg.norm(vec) + 1e-8)  # L2-normalise

        embeddings.append({
            "identity":  img_path,
            "embedding": vec.tolist()
        })

    elapsed = time.time() - start
    eta = (elapsed / (idx + 1)) * (total - idx - 1)
    print(f"[{idx+1}/{total}]  faces so far: {len(embeddings):4d}  |  ETA {eta:.0f}s", end="\r")

print(f"\n\nDone! Indexed {len(embeddings)} face embeddings from {total} photos.")

with open(PKL_FILE, "wb") as f:
    pickle.dump(embeddings, f)

print(f"Saved -> {PKL_FILE}")
print(f"This file MUST be uploaded to Render inside the photos_db/ folder.")
