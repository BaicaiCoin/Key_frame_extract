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

    def merge_areas(self, contours, merged_threshold):
        """
        Aggregates a set of rectangles that are close together and returns a list of the smallest rectangles of the settlement.
        
        参数:
            rectangles (list): Enter a list of rectangles, each of which is a 2x2 matrix.
            distance_threshold (int): The threshold for determining whether a rectangle is "close."
        """
        def calculate_distance(rect1, rect2):
            x1_A, y1_A, x2_A, y2_A = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1]
            x1_B, y1_B, x2_B, y2_B = rect2[0][0], rect2[0][1], rect2[1][0], rect2[1][1]

            dx = max(0, max(x1_B - x2_A, x1_A - x2_B))
            dy = max(0, max(y1_B - y2_A, y1_A - y2_B))

            return dx + dy

        n = len(contours)
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                if rank[root_x] > rank[root_y]:
                    parent[root_y] = root_x
                elif rank[root_x] < rank[root_y]:
                    parent[root_x] = root_y
                else:
                    parent[root_y] = root_x
                    rank[root_x] += 1

        for i in range(n):
            for j in range(i + 1, n):
                if calculate_distance(contours[i], contours[j]) <= merged_threshold:
                    union(i, j)

        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(contours[i])

        merged_rectangles = []
        for group in groups.values():
            min_x = min(rect[0][0] for rect in group)
            min_y = min(rect[0][1] for rect in group)
            max_x = max(rect[1][0] for rect in group)
            max_y = max(rect[1][1] for rect in group)
            merged_rectangles.append([[min_x, min_y], [max_x, max_y]])

        return merged_rectangles

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

        diff_threshold = 10
        area_threshold = int(height * 2)
        total_area_threshold = int(width * 3)
        merged_threshold = int(height / 7)
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

            total_area = 0
            for contour in diff_contours:
                area = cv2.contourArea(contour)
                total_area += area

            if total_area > total_area_threshold:
                rectangles = [cv2.boundingRect(contour) for contour in diff_contours]
                rectangles = [[[x, y], [x + w, y + h]] for x, y, w, h in rectangles]

                merged_areas = self.merge_areas(rectangles, merged_threshold)

                for area in merged_areas:
                    x1, y1 = area[0]
                    x2, y2 = area[1]
                    if (x2 - x1) * (y2 - y1) > area_threshold:
                        self.draw_highlighted_rectangle(curr_frame, x1, y1, x2, y2)
                print(f"frame {frame_idx}: total area of change {total_area}, key frame detected")
            else:
                print(f"frame {frame_idx}: total area of change {total_area}, key frame not detected")

            output_path = os.path.join(output_dir, f"frame_{frame_idx}.png")
            cv2.imwrite(output_path, curr_frame)
            print(f"save frame {frame_idx} to {output_path}")

            prev_gray = curr_gray
            frame_idx += skip_frames + 1

        cap.release()
        print(f"all frames save to {output_dir}")

    def extract_key_frames(self, interval=1):
        print("Start processing video: ",self.video_path)
        self.extract_by_frame_diff()
        
        