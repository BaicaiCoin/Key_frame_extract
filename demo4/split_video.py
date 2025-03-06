import math
import os
import shutil
import subprocess

class SplitVideo:
    def __init__(self):
        pass

    @staticmethod
    def split_video():
        segment_length = 240
        segment_max_length = 300

        for file in os.listdir("video"):
            file_name = file.split("/")[-1].split(".")[0]
            if os.path.exists(f"results/sub_video/{file_name}_0.mp4"):
                continue
            get_total_cmd = [
                "ffprobe",
                "-i", f"video/{file}",
                "-show_entries", "format=duration",
                "-v", "quiet",
                "-of", "csv=p=0"
            ]
            result = subprocess.run(get_total_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            total_duration = float(result.stdout.strip())
            if total_duration <= segment_max_length:
                shutil.copy(f"video/{file}", f"results/sub_video/{file_name}_0.mp4")

            segment_num = math.ceil(total_duration / segment_length)
            last_segment_length = total_duration - (segment_num - 1) * segment_length

            if last_segment_length < segment_max_length - segment_length:
                segment_num -= 1
            
            output_path = os.path.join("results/sub_video", f"{file_name}_%d.mp4")
            cut_cmd = [
                "ffmpeg",
                "-i", f"video/{file}",
                "-c", "copy",
                "-map", "0",
                "-f", "segment",
                "-segment_times",
                ",".join([str(segment_length * i) for i in range(1, segment_num)]),
                "-reset_timestamps", "1",
                output_path
            ]
            subprocess.run(cut_cmd)