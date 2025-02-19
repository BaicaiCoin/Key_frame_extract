from openai import OpenAI
import whisper_timestamped as whisper
import base64
import markdown
import ffmpeg
import os
import json
import math

from prompt import Prompt

class KeyFrameExtract:

    whisper_model = whisper.load_model("small", device="cpu")
    client = OpenAI()

    def __init__(self, video_path=None):
        self.video_path = video_path
        self.file_name = video_path.split("/")[-1].split(".")[0]
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def extract_frames(self, output_folder, interval=1):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        ffmpeg.input(self.video_path, ss=0).output(
            os.path.join(output_folder, "frame_%05d.jpg"), 
            vf=f"fps=1/{interval}", 
            start_number=0
        ).run()
    
    def extract_subtitles(self, output_folder):
        output_audio = f"{output_folder}/{self.file_name}.wav"
        ffmpeg.input(self.video_path).output(output_audio).run()
        audio = whisper.load_audio(output_audio)
        subtitles = whisper.transcribe(KeyFrameExtract.whisper_model, audio, language="en")

        segments_data = []
        for segment in subtitles["segments"]:
            segments_data.append({
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"]
            })
        with open(output_audio.replace(".wav", ".json"), "w", encoding="utf-8") as json_file:
            json.dump(segments_data, json_file, indent=2, ensure_ascii=False)

        return subtitles

    def gpt_input_generate(self, subtitles, interval=1):
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
                image_code = self.encode_image(f"results/frames/{self.file_name}/frame_{j:05d}.jpg")
                content.append({
                    "type":"image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{image_code}",
                        # "url": f"data:image/jpeg;base64,{j:05d}",
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
    
    def extract_key_frames(self, interval=1):
        print("Start processing video: ",self.video_path)
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
        key_frame_result = markdown.markdown(response.choices[0].message.content)
        with open(f"results/key_frames/{self.file_name}.md", "w", encoding="utf-8") as file:
            file.write(key_frame_result)

if __name__ == "__main__":
    for root, dirs, files in os.walk("video"):
        for file in files:
            key_frame = KeyFrameExtract(f"video/{file}")
            key_frame.extract_key_frames(2)