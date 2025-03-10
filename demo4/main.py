import os

from key_frame_extract import KeyFrameExtract
from split_video import SplitVideo

if __name__ == "__main__":
    SplitVideo.split_video()
    for root, dirs, files in os.walk("results/sub_video"):
        files = sorted(files)
        for file in files:
            # key_frame = KeyFrameExtract(f"results/sub_video/{file}")
            key_frame = KeyFrameExtract(f"results/sub_video/example2_0.mp4")
            key_frame.extract_key_frames(2)
            break