import os

from key_frame_extract import KeyFrameExtract

if __name__ == "__main__":
    for root, dirs, files in os.walk("video"):
        for file in files:
            key_frame = KeyFrameExtract(f"video/{file}")
            key_frame.extract_key_frames(2)
            break