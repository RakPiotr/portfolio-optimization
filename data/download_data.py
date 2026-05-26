from pathlib import Path
import shutil
import kagglehub

# Folder where this Python file is located
current_folder = Path(__file__).resolve().parent

# Kaggle dataset slug
dataset_slug = "jacksaleeby/s-and-p500-historical-data"

# Download dataset to KaggleHub cache
downloaded_path = Path(kagglehub.dataset_download(dataset_slug))

# Destination folder beside this script
destination = current_folder

# Remove old copy if it exists
if destination.exists():
    shutil.rmtree(destination)

# Copy downloaded dataset into this script's folder
shutil.copytree(downloaded_path, destination)

print(f"Dataset saved to: {destination}")