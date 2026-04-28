import os
import urllib.request
import tarfile
from pathlib import Path

def download_bows2():
    url = "http://bows2.ec-lille.fr/BOWS2OrigEp3.tgz"
    target_dir = Path("data/BOWS2")
    cover_dir = target_dir / "cover"
    tar_path = target_dir / "BOWS2OrigEp3.tgz"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    cover_dir.mkdir(parents=True, exist_ok=True)
    
    if len(list(cover_dir.glob("*.pgm"))) >= 10000:
        print("BOWS2 dataset already exists!")
        return

    print("Downloading BOWS2 dataset (this may take a while)...")
    if not tar_path.exists():
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")

    print("Extracting BOWS2...")
    with tarfile.open(tar_path, "r:gz") as tar:
        # Extract everything
        tar.extractall(path=target_dir)
        
    # The tarball extracts to data/BOWS2/BOWS2OrigEp3/*.pgm
    # Move them to data/BOWS2/cover
    extracted_folder = target_dir / "BOWS2OrigEp3"
    if extracted_folder.exists():
        for file in extracted_folder.glob("*.pgm"):
            file.rename(cover_dir / file.name)
        extracted_folder.rmdir()
        
    print(f"BOWS2 successfully extracted to {cover_dir}")

if __name__ == "__main__":
    download_bows2()
