"""cleanup_images.py - script to normalize and resize images scraped from ThisPersonDoesNotExist.com."""

# NOTE: you must run get_fake_images.py AND get_real_images.py before running this script

# Install dependencies
import logging
import os
from PIL import Image

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Resize and reformat images
def resize_images(source_dir, target_size=(128, 128)):
    logging.info(f"Starting image cleanup in: {source_dir}")
    count = 0
    failed = 0

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                path = os.path.join(root, file)
                try:
                    # convert all pics to RGB
                    img = Image.open(path).convert("RGB")
                    img = img.resize(target_size)
                    img.save(path)
                    count += 1
                    if count % 50 == 0:
                        logging.warning(f"Failed to process {file}: {e}")
                except Exception as e:
                    logging.warning(f"Failed to process {file}: {e}")
                    failed += 1
    
    logging.info(f"Finished resizing images in: {source_dir}")
    logging.info(f"Total images processed: {count}")
    if failed:
        logging.warning(f"Failed to process {failed} images.")

resize_images("image_data/real")
resize_images("image_data/fake")
