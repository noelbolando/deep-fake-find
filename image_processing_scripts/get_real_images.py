"""get_real_images.py - script to scrape fake images from ThisPersonDoesNotExist.com."""

# NOTE: you must manually download a zip of the photos at: https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
# After which you can run this script (origin destination: ~/Downloads)

# Install dependencies
import os

def rename_images(directory, prefix="real"):
    # Get all image files
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    script_space = "image_data/real"

    for i, filename in enumerate(image_files):
        ext = os.path.splitext(filename)[1]
        new_name = f"{prefix}_{i}{ext}"
        src = os.path.join(directory, filename)
        dst = os.path.join(script_space, new_name)

        try:
            os.rename(src, dst)
            print(f"Renamed: {filename} -> {new_name}")
        except Exception as e:
            print(f"Failed to rename {filename}: {e}")

# Do it
rename_images("/Users/noelboland/Downloads/archive/", prefix="real")
