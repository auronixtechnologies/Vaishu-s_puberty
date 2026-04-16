import os
import cv2
import pickle
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

PHOTOS_DIR = r"D:\Vaishu\Photos"
OUTPUT_PKL = "lightweight_embeddings.pkl"


# 🔥 SAME embedding logic as backend (IMPORTANT)
def get_embedding(img_bgr):
    img = cv2.resize(img_bgr, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    emb = img.flatten().astype("float32")
    emb = emb / np.linalg.norm(emb)
    return emb


def index_photos():
    if not os.path.exists(PHOTOS_DIR):
        logging.error(f"Directory not found: {PHOTOS_DIR}")
        return

    all_files = [f for f in os.listdir(PHOTOS_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not all_files:
        logging.info("No photos found.")
        return

    representations = []

    logging.info("Starting lightweight indexing (NO DeepFace)...")

    for file in all_files:
        path = os.path.join(PHOTOS_DIR, file)

        try:
            img = cv2.imread(path)
            if img is None:
                logging.warning(f"Skipping unreadable image: {file}")
                continue

            embedding = get_embedding(img)

            representations.append({
                "identity": path,
                "embedding": embedding.tolist()
            })

            logging.info(f"Indexed: {file}")

        except Exception as e:
            logging.error(f"Error processing {file}: {e}")

    # Save .pkl
    output_path = os.path.join(PHOTOS_DIR, OUTPUT_PKL)

    with open(output_path, "wb") as f:
        pickle.dump(representations, f)

    logging.info(f"✅ Indexing complete. Saved to {output_path}")


if __name__ == "__main__":
    index_photos()
