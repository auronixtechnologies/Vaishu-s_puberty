import os
import cv2
import logging
from deepface import DeepFace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

PHOTOS_DIR = r"D:\Vaishu\Photos"

def index_photos():
    if not os.path.exists(PHOTOS_DIR):
        logging.error(f"Directory not found: {PHOTOS_DIR}")
        return
    
    all_files = [f for f in os.listdir(PHOTOS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not all_files:
        logging.info("No photos found.")
        return
        
    dummy_path = os.path.join(PHOTOS_DIR, all_files[0])
    
    logging.info("Running Advanced ML ArcFace Indexer... This handles threshold calibration naturally!")
    logging.info("DeepFace is scaling the directory into standard pkl models. Please wait a few minutes...")
    
    # This forces DeepFace to analyze every single photo in the directory 
    # and serialize the exact math distances into its native C-cache '.pkl' file.
    try:
        DeepFace.find(img_path=dummy_path, db_path=PHOTOS_DIR, model_name="ArcFace", enforce_detection=False)
        logging.info("Successfully generated generic native cache! (representations_arcface.pkl)")
    except Exception as e:
        logging.error(f"Error during indexing: {e}")

if __name__ == "__main__":
    # Force delete the generic cache so it physically builds a fresh true copy
    pkl_path = os.path.join(PHOTOS_DIR, "representations_arcface.pkl")
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
        
    index_photos()