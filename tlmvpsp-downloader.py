import requests
from bs4 import BeautifulSoup
import yt_dlp
import re
import numpy as np
import time

def get_output_name(url):
    regex = r"emission-du-([\w-]+)\.html"
    match = re.search(regex, url)
    if match:
        return f'{match.group(1)}.mp4'
    else:
        return 'unknown-date.mp4'
    
def get_video_urls(source, pattern):
    req = requests.get(source)
    soup = BeautifulSoup(req.text, 'html.parser')
    
    # Collect video URLs that match the pattern "emission-du"
    urls = [f'{pattern+a["href"]}' for a in soup.find_all('a', href=True) if "emission-du" in a['href']]
    return urls

# Recherche des liens d'émissions

def download_video(video_url):
    options = {
        'format': 'hls-2218',  # Choose the appropriate format
        'outtmpl': 'E:\Emissions\\' + get_output_name(video_url)  # Save to your desired location
    }
    
    # Download the video using yt_dlp
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([video_url])

"""
format_id: dash-video=400000, ext: mp4, resolution: 384x216, note: DASH video
format_id: hls-522, ext: mp4, resolution: 384x216, note: N/A
format_id: dash-video=950000, ext: mp4, resolution: 640x360, note: DASH video
format_id: hls-1105, ext: mp4, resolution: 640x360, note: N/A
format_id: dash-video=1400000, ext: mp4, resolution: 960x540, note: DASH video
format_id: hls-1582, ext: mp4, resolution: 960x540, note: N/A
format_id: dash-video=2000000, ext: mp4, resolution: 1280x720, note: DASH video
format_id: hls-2218, ext: mp4, resolution: 1280x720, note: N/A
format_id: dash-video=5000000, ext: mp4, resolution: 1920x1080, note: DASH video
format_id: hls-5398, ext: mp4, resolution: 1920x1080, note: N/A
"""

# def main():
    # seen_urls = set(get_video_urls())  # To store URLs that have already been downloaded
    
    # while True:
    #     # Get the current list of video URLs
    #     current_urls = set(get_video_urls())
    #     print(current_urls)
    #     # Find new URLs that have not been seen before
    #     new_urls = current_urls - seen_urls
        
    #     if new_urls:
    #         first_new_url = next(iter(new_urls))  # Get the first new URL
    #         print(f"Found a new video: {first_new_url}. Downloading...")
    #         download_video(first_new_url)
        
    #     # Update the list of seen URLs
    #     seen_urls=current_urls
        
    #     # Wait for 1 hour before checking again (3600 seconds)
    #     time.sleep(3600)
