import re
from openai import OpenAI
import whisper_timestamped as whisper
import base64
import markdown
import ffmpeg
import os
import json
import cv2
import numpy as np

from actions_extract import ActionsExtract
from key_frame_prompt import Prompt

class KeyFrameExtract:

    whisper_model = whisper.load_model("small", device="cpu")
    client = OpenAI()

    def __init__(self, video_path=None):
        self.video_path = video_path
        self.file_name = video_path.split("/")[-1].split(".")[0]

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def merge_areas(self, contours, merged_threshold):
        """
        Aggregates a set of rectangles that are close together and returns a list of the smallest rectangles of the settlement.
        
        Params:
            contours (list): a list of rectangles, each of which is a 2x2 matrix.
            merged_threshold (int): The threshold for determining whether rectangles are close.
        """
        def calculate_distance(rect1, rect2):
            """
            Calculates the nearest boundary distance between two rectangles.
            """
            x1_A, y1_A, x2_A, y2_A = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1]
            x1_B, y1_B, x2_B, y2_B = rect2[0][0], rect2[0][1], rect2[1][0], rect2[1][1]
            dx = max(0, max(x1_B - x2_A, x1_A - x2_B))
            dy = max(0, max(y1_B - y2_A, y1_A - y2_B))
            return dx + dy

        n = len(contours)
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            """Finds the root node of the set"""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            """Merge two sets"""
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
    
    def add_frame_number(self, frame, frame_idx):
        """
        Add the frame number to the bottom-center of the frame.
        """
        text = f"Frame: {frame_idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (255, 255, 255)
        text_bg_color = (0, 0, 0)

        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_width, text_height = text_size

        text_x = int(frame.shape[1] * 4 / 9)
        text_y = frame.shape[0] - 5

        cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), text_bg_color, -1)

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    def draw_highlighted_rectangle(self, frame, min_x, min_y, max_x, max_y, number):
        """Draw and highlight the area of change"""
        padding = 15
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
        frame[min_y:max_y, min_x:max_x] = cv2.addWeighted(
            frame[min_y:max_y, min_x:max_x], 0.9,
            np.full_like(frame[min_y:max_y, min_x:max_x], (0, 0, 255)), 0.1, 0
        )

        text = str(number)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        text_color = (255, 255, 255)
        text_bg_color = (0, 0, 255)
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_width, text_height = text_size
        text_x = max(0, min_x + 5)
        text_y = max(0, min_y + text_height + 5)

        cv2.rectangle(frame, (text_x - 2, text_y - text_height - 2), (text_x + text_width + 2, text_y + 2), text_bg_color, -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    def add_image_to_input(self, content, num, is_curr):
        if is_curr:
            path = f"results/diffs/{self.file_name}/frame_{num}_labelled.png"
        else:
            if not os.path.exists(f"results/marked_frames/{self.file_name}"):
                os.mkdir(f"results/marked_frames/{self.file_name}")
            path = f"results/marked_frames/{self.file_name}/{num}.png"
            if not os.path.exists(path):
                source_path = f"results/frames/{self.file_name}/frame_{num}.png"
                image = cv2.imread(source_path).copy()
                self.add_frame_number(image, num)
                cv2.imwrite(path, image)

        image_code = self.encode_image(path)
        content.append({
            "type":"image_url",
            "image_url":{
                "url": f"data:image/jpeg;base64,{image_code}",
                # "url": f"data:image/jpeg;base64,{path}",
                "detail":"low"
            }
        })

    def filter_by_gpt(self, curr, prev, next, prev_key, action, fps):
        print(f"Ask GPT frame {curr}, action: {action}")
        content = []
        prompts = Prompt()
        
        with open(f"results/subtitles/{self.file_name}.json", "r", encoding="utf-8") as json_file:
            subtitles = json.load(json_file)
        segments = subtitles["segments"]
        sentence = ""
        tokens = [{
                "name":"<Previous Key Frame>",
                "time":float(prev_key) / float(fps)
        }]
        for i in range(len(prev)):
            tokens.append({
                "name":f"<Previous Sampled Frame {i}>",
                "time":float(prev[i]) / float(fps)
            })
        tokens.append({
            "name":"<Current Frame>",
            "time":float(curr) / float(fps)
        })
        for i in range(len(next)):
            tokens.append({
                "name":f"<Next Sampled Frame {i}>",
                "time":float(next[i]) / float(fps)
            })
        words = []
        start_time = tokens[0]["time"]
        end_time = tokens[len(tokens) - 1]["time"]
        for i in range(0, len(segments)):
            if segments[i]["end"] >= start_time:
                for j in range(0, len(segments) - i):
                    if not segments[i + j]["start"] > end_time:
                        words += segments[i + j]["words"]
                    else:
                        break
                break
        while(len(tokens) > 0 or len(words) > 0):
            if len(words) == 0 or (len(tokens) != 0 and tokens[0]["time"] <= words[0]["start"]):
                sentence += f"{tokens[0]['name']} "
                del(tokens[0])
            else:
                sentence += f"{words[0]['text']} "
                del(words[0])
        text_data = f"""
{{
    "Current Frame": "{curr}",
    "Previous Key Frame": "{prev_key}",
    "Previous Sampled Frames": "{prev}",
    "Next Sampled Frames": "{next}",
    "Nearby Subtitles": "{sentence}",
    "Action to Be Completed": "{action}"
}}
"""

        self.add_image_to_input(content, prev_key, False)
        for i in range(len(prev)):
            self.add_image_to_input(content, prev[i], False)
        self.add_image_to_input(content, curr, True)
        for i in range(len(next)):
            self.add_image_to_input(content, next[i], False)
        content.append({
            "type":"text",
            "text": prompts.get_prompts()
        })
        content.append({
            "type":"text",
            "text":"[data start]"
        })
        content.append({
            "type":"text",
            "text":text_data
        })
        content.append({
            "type":"text",
            "text":"[data end]"
        })
        # with open("input.txt", "a", encoding="utf-8") as file:
        #     file.write(json.dumps(content, ensure_ascii=False, indent=4))
        #     file.write("\n\n")
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role":"user",
                    "content":content
                }
            ]
        )
        result = markdown.markdown(response.choices[0].message.content)
        code_block_match = re.search(r"<code.*?>(.*?)</code>", result, re.DOTALL)
        code_block_content = code_block_match.group(1).strip()
        if code_block_content.startswith("json"):
            code_block_content = code_block_content[4:].strip()
        result_json = json.loads(code_block_content)
        with open("output.txt", "a", encoding="utf-8") as file:
            file.write(json.dumps(result_json, ensure_ascii=False, indent=4))
            file.write("\n\n")
        return result_json

    def extract_by_frame_diff(self):

        key_frames = []

        """Initially filter key frames by inter-frame differences and locate the area of change"""
        output_dir = f"results/diffs/{self.file_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Find the last frame
        max_frame_num = -1
        for file_name in os.listdir(f"results/frames/{self.file_name}"):
            match = re.match(r"frame_(\d+)\.png", file_name)
            if match:
                num = int(match.group(1))
                if num > max_frame_num:
                    max_frame_num = num
        print(max_frame_num)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video file: {self.video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        diff_threshold = 20
        area_threshold = int(height * 2)
        total_area_threshold = int(width * 3)
        merged_threshold = int(height / 7)
        skip_frames = fps - 1

        with open(f"results/actions/{self.file_name}.json", "r", encoding="utf-8") as f:
            actions = json.load(f)

        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("The video frame cannot be read")
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_idx = fps
        prev_sample_frame = [0]
        prev_key_frame = 0

        # Iteration of each sampled frame
        while len(actions) > 0:
            # Gain curr_frame
            is_end = False
            for _ in range(skip_frames):
                ret = cap.grab()
                if not ret:
                    is_end = True
                    break
            if is_end:
                break
            ret, curr_frame = cap.read()
            if not ret:
                break

            next_sample_frame = [frame_idx + fps, frame_idx + fps * 2]
            for i in range(2):
                if next_sample_frame[0] > max_frame_num:
                    del(next_sample_frame[0])

            # Calculate the difference between frames
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            _, diff_thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            diff_contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate the total area of change
            total_area = 0
            for contour in diff_contours:
                area = cv2.contourArea(contour)
                total_area += area

            # Merge change areas
            if total_area > total_area_threshold:
                rectangles = [cv2.boundingRect(contour) for contour in diff_contours]
                rectangles = [[[x, y], [x + w, y + h]] for x, y, w, h in rectangles]
                merged_areas = self.merge_areas(rectangles, merged_threshold)

                # Draw rectangles
                number = 1
                for area in merged_areas:
                    x1, y1 = area[0]
                    x2, y2 = area[1]
                    if (x2 - x1) * (y2 - y1) > area_threshold:
                        self.draw_highlighted_rectangle(curr_frame, x1, y1, x2, y2, number)
                        number += 1
                self.add_frame_number(curr_frame, frame_idx)
                cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx}_labelled.png"), curr_frame)

                prev_gray = curr_gray
                # print(f"Frame {frame_idx}: Total area of change {total_area}, the key frame is detected")

                # Give the picture to gpt for judgment
                gpt_result = self.filter_by_gpt(frame_idx, prev_sample_frame, next_sample_frame, prev_key_frame, actions[0], fps)
                while gpt_result["action_completed_early"] == 1:
                    key_frames[len(key_frames) - 1]["actions"] += actions[0]
                    del(actions[0])
                    gpt_result = self.filter_by_gpt(frame_idx, prev_sample_frame, next_sample_frame, prev_key_frame, actions[0], fps)
                if gpt_result["key_frame"] == 1:
                    key_frames.append({
                        "frame": frame_idx,
                        "actions": gpt_result["action"]
                    })
                    prev_key_frame = frame_idx
                    del(actions[0])

                prev_sample_frame.append(frame_idx)
                if len(prev_sample_frame) > 2:
                    del(prev_sample_frame[0])
            # else:
                # print(f"Frame {frame_idx}: the key frame threshold is not reached")

            frame_idx += skip_frames + 1

        cap.release()

        with open(f"results/key_frames/{self.file_name}.json", "w", encoding="utf-8") as json_file:
            json.dump(key_frames, json_file, indent=2, ensure_ascii=False)

    def extract_key_frames(self, interval=2):
        print("Start processing video: ",self.video_path)
        actions_extract = ActionsExtract(self.video_path)
        actions_extract.extract_actions()
        self.extract_by_frame_diff()
        
        