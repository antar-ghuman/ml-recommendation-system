import os
import urllib.request
import zipfile
from pathlib import Path

def download_movielens():
    """Download MovieLens 25M dataset"""
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    zip_path = data_dir / "ml-25m.zip"
    
    print("Downloading MovieLens 25M dataset...")
    print("This may take 5-10 minutes (250MB file)")
    
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print("Cleaning up...")
    os.remove(zip_path)
    
    print("✅ Dataset ready in data/raw/ml-25m/")
    print(f"   - ratings.csv: 25M ratings")
    print(f"   - movies.csv: 62K movies")
    print(f"   - tags.csv: 1M tags")

if __name__ == "__main__":
    download_movielens()