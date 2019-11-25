import json
from google_images_download import google_images_download

def download_images():
    google_images_downloader = google_images_download.googleimagesdownload()
    with open('images_config.json') as config_file:
        config_data = json.load(config_file)
        for record in config_data['Records']:
            google_images_downloader.download(record)