from openai import OpenAI
import whisper_timestamped as whisper
import base64
import markdown
import ffmpeg
import os
import json
import cv2
import numpy as np

class KeyFrameExtract:

    whisper_model = whisper.load_model("small", device="cpu")
    client = OpenAI()

    def __init__(self, video_path=None):
        self.video_path = video_path
        self.file_name = video_path.split("/")[-1].split(".")[0]

    def draw_highlighted_rectangle(self, frame, min_x, min_y, max_x, max_y):
        """Draw and highlight the area of change"""
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        frame[min_y:max_y, min_x:max_x] = cv2.addWeighted(
            frame[min_y:max_y, min_x:max_x], 0.5,
            np.full_like(frame[min_y:max_y, min_x:max_x], (0, 0, 255)), 0.5, 0
        )

    def extract_by_frame_diff(self):
        """Initially filter key frames by inter-frame differences and locate the area of change"""
        output_dir = f"results/diffs/{self.file_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video file: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        """
        diff_threshold (int): The pixel difference threshold, beyond which pixels are considered to have changed.
        area_threshold (int): Minimum area threshold for a single area of change.
        total_area_threshold (int): The sum threshold of the areas of all changing regions, beyond which the key frame is considered.
        skip_frames (int): The number of frames jumped. The difference is calculated every skip_frames frame.
        """
        diff_threshold = 10
        area_threshold = int(height / 20)
        total_area_threshold = int(width)
        skip_frames = int(fps) - 1

        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("The video frame cannot be read")
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_idx = 0
        while True:
            for _ in range(skip_frames):
                ret = cap.grab()
                if not ret:
                    break
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            _, diff_thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            diff_contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
            total_area = 0  # 总的变化区域面积
            for contour in diff_contours:
                area = cv2.contourArea(contour)
                total_area += area  # 累加总面积
                if area > area_threshold:  # 只过滤掉非常小的噪声区域
                    x, y, w, h = cv2.boundingRect(contour)
                    # 更新聚合矩形的边界
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)

            # 如果总变化面积超过阈值，认为是关键帧
            if total_area > total_area_threshold:
                # 绘制聚合的大矩形
                self.draw_highlighted_rectangle(curr_frame, min_x, min_y, max_x, max_y)
                print(f"帧 {frame_idx}: 总变化面积 {total_area}，检测到关键帧，变化区域 ({min_x}, {min_y}, {max_x}, {max_y})")
            else:
                print(f"帧 {frame_idx}: 总变化面积 {total_area}，未达到关键帧阈值")

            # 保存当前帧为图片
            output_path = os.path.join(output_dir, f"frame_{frame_idx}.png")
            cv2.imwrite(output_path, curr_frame)
            print(f"保存帧 {frame_idx} 到 {output_path}")

            # 更新上一帧
            prev_gray = curr_gray
            frame_idx += skip_frames + 1

        cap.release()
        print(f"所有帧已保存到 {output_dir}")
    
    def extract_key_frames(self, interval=1):
        print("Start processing video: ",self.video_path)
        self.extract_by_frame_diff()
        
        