import ast
import re
from openai import OpenAI
import whisper_timestamped as whisper
import base64
import ffmpeg
import os
import json
import markdown

from actions_extract_prompt import Prompt

class ActionsExtract:

    whisper_model = whisper.load_model("small", device="cpu")
    client = OpenAI()

    def __init__(self, video_path=None):
        self.video_path = video_path
        self.file_name = video_path.split("/")[-1].split(".")[0]
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def extract_frames(self, output_folder, interval=4):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            ffmpeg.input(self.video_path, ss=0).output(
                os.path.join(output_folder, "frame_%d.jpg"), 
                vf=f"fps=1/{interval}", 
                start_number=0
            ).run()
    
    def extract_subtitles(self, output_folder):
        output_audio = f"{output_folder}/{self.file_name}.wav"
        if not os.path.exists(output_audio):
            ffmpeg.input(self.video_path).output(output_audio).run()
            audio = whisper.load_audio(output_audio)
            subtitles = whisper.transcribe(ActionsExtract.whisper_model, audio, language="en")
            with open(output_audio.replace(".wav", ".json"), "w", encoding="utf-8") as json_file:
                json.dump(subtitles, json_file, indent=2, ensure_ascii=False)

            segments_data = []
            for segment in subtitles["segments"]:
                segments_data.append({
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": segment["text"]
                })
            with open(output_audio.replace(".wav", "_brief.json"), "w", encoding="utf-8") as json_file:
                json.dump(segments_data, json_file, indent=2, ensure_ascii=False)

        else:
            with open(output_audio.replace(".wav", ".json"), "r", encoding="utf-8") as json_file:
                subtitles = json.load(json_file)

        return subtitles

    def gpt_input_generate(self, subtitles, interval=4):
        content = []
        segments = subtitles["segments"]
        images = sorted(os.listdir(f"results/frames/{self.file_name}"))
        image_num = 0
        image_curr_num = 0
        prompts = Prompt(interval)

        content.append({
            "type":"text",
            "text": prompts.get_prompts()
        })
        content.append({
            "type":"text",
            "text":"[video start]"
        })
        for i in range(0, len(segments), 3):
            start_time = segments[i]["start"]
            if i + 3 < len(segments):
                end_time = segments[i + 3]["start"]
            else: 
                end_time = segments[len(segments) - 1]["end"]
            
            sentence = ""
            words = []
            for j in range(3):
                words += segments[i + j]["words"]
            while image_curr_num * interval < end_time:
                if len(words) <= 0 or image_curr_num * interval < words[0]["start"]:
                    sentence += f"<cs{image_curr_num:05d}> "
                    image_curr_num += 1
                else:
                    sentence += f"{words[0]['text']} "
                    del(words[0])
            
            for j in range(image_num, image_curr_num):
                image_code = self.encode_image(f"results/frames/{self.file_name}/frame_{j}.jpg")
                content.append({
                    "type":"image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{image_code}",
                        # "url": f"data:image/jpeg;base64,{j}",
                        "detail":"low"
                    }
                })
            content.append({
                "type":"text",
                "text":sentence
            })

            image_num = image_curr_num

        content.append({
            "type":"text",
            "text":"[video end]"
        })
        return content
    
    def extract_actions(self, interval=4):
        if not os.path.exists(f"results/actions/{self.file_name}.json"):
            self.extract_frames(f"results/frames/{self.file_name}", interval)
            subtitles = self.extract_subtitles("results/subtitles")
            content = self.gpt_input_generate(subtitles, interval)

            # with open("input.json", "w") as f:
            #     json.dump(content, f, indent=4, ensure_ascii=False)

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role":"user",
                        "content":content
                    }
                ]
            )
            actions_result = markdown.markdown(response.choices[0].message.content)
            code_block_match = re.search(r"<code.*?>(.*?)</code>", actions_result, re.DOTALL)
            if code_block_match:
                code_block_content = code_block_match.group(1).strip()
                if code_block_content.startswith("json"):
                    code_block_content = code_block_content[4:].strip()
                try:
                    result_json = json.loads(code_block_content)
                except json.JSONDecodeError as e:
                    print("JSON parse failed", e)
            else:
                print("can't find <code>")
            with open(f"results/actions/{self.file_name}.json", "w", encoding="utf-8") as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)