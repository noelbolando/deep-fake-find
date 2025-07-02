"""get_fake_images.py - script to scrape fake images from ThisPersonDoesNotExist.com."""

# Install dependencies
import requests
import os

output_dir = "image_data/fake"
os.makedirs(output_dir, exist_ok=True)

for i in range(500):
    try:
        response = requests.get("https://thispersondoesnotexist.com", timeout=5)
        if response.status_code == 200:
            with open(f"{output_dir}/fake_{i}.jpg", "wb") as f:
                f.write(response.content)
            print(f"Downloaded fake_{i}.jpg")
    except Exception as e:
        print(f"Error at {i}: {e}")
