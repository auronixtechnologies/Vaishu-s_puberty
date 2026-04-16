import json
import os
import requests

def list_drive_folder(folder_id, api_key):
    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        'q': f"'{folder_id}' in parents and trashed=false",
        'key': api_key,
        'fields': 'files(id, name)',
        'pageSize': 1000
    }
    
    print(f"Fetching photos from Google Drive folder: {folder_id}...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        files = response.json().get('files', [])
        return files
    else:
        print(f"Error fetching data: {response.text}")
        return []

if __name__ == "__main__":
    print("-" * 50)
    print("Google Drive Sync Script")
    print("-" * 50)
    
    FOLDER_URL = input("Enter your public Google Drive Folder Link: ").strip()
    API_KEY = input("Enter your Google Cloud API Key: ").strip()
    
    # Extract folder ID from URL
    # https://drive.google.com/drive/folders/1ckzwzMFuWd2kIVX1xKnpx-PKRzVD8aMv?usp=drive_link
    try:
        if "folders/" in FOLDER_URL:
            folder_id = FOLDER_URL.split("folders/")[1].split("?")[0]
        else:
            folder_id = FOLDER_URL
    except:
        folder_id = FOLDER_URL
        
    if not API_KEY or not folder_id:
        print("Folder URL and API Key are required!")
        exit(1)
        
    files = list_drive_folder(folder_id, API_KEY)
    
    if not files:
        print("No files discovered or the folder is not public!")
        exit(1)
        
    print(f"Discovered {len(files)} files in Google Drive!")
    
    drive_map = {}
    for f in files:
        # e.g., IMG_2385.JPG -> 1ckzwzMFuWd...
        drive_map[f['name']] = f['id']
        
    out_file = "drive_links.json"
    with open(out_file, "w", encoding="utf-8") as out:
        json.dump(drive_map, out, indent=4)
        
    print(f"Successfully generated {out_file} with {len(drive_map)} entries!")
    print("The backend is now ready to serve these images from the cloud!")
