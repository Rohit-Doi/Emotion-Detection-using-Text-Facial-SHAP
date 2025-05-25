import requests
import os

# Create directory if it doesn't exist
os.makedirs('data/full_dataset', exist_ok=True)

# URLs and corresponding file paths
urls = [
    'https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv',
    'https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv',
    'https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv'
]
file_paths = [
    'data/full_dataset/goemotions_1.csv',
    'data/full_dataset/goemotions_2.csv',
    'data/full_dataset/goemotions_3.csv'
]

# Download each file
for url, file_path in zip(urls, file_paths):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {file_path}")
    else:
        print(f"Failed to download {url}")