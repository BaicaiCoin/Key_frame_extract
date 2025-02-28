class Prompt:

    def __init__(self, interval):
        self.interval = interval

    def outset(self) -> str:
        return """
You are a data annotator, and you will receive video clips sourced from YouTube.

**VIDEO DETAILS**
The video clips consist of tutorials on various computer applications, presented through screen recordings of the operations, accompanied by voiceovers or subtitles.

**TASK DETAILS**
Your task is to summarize all user interactions from the video in sequential order and output them.
Examples of user interactions:
-Clicking a interactive element with the left or right mouse button
-Entering a complete text
-Pressing a keyboard shortcut
When a subsequent interaction depends on the result of a previous interaction that causes a page change, the operations must not be merged.
For example:
-Right-clicking a file and selecting "Properties" should be divided into two separate interactions, as the "Properties" button can only be located after right-clicking the file.
"""

    def images_input_format(self) -> str:
        prompt = f"""
**INPUT FORMAT**
1. Due to the limitations of the input modality, the data consists of a combination of images and text, rather than the video itself.
2. The image-text combination starts with the [video start] tag and ends with the [video end] tag.
3. Images are screenshots taken every {self.interval} second from the video, while the text corresponds to the video's narration subtitles. You are required to infer the content of the video between these screenshots based on the time gaps. 
   When selecting a keyframe, choose the next available screenshot after a period of time following the execution of the unit action, when the page has stabilized.
4. The presentation format for the image-text combination follows this pattern: (all screenshots from a specific time period in the video, arranged in order; three lines of subtitles spoken during that period), and this cycle repeats until the video ends.
5. Subtitles will contain tags like "<cs00001>", "<cs00002>", "<cs00003>", with the format "<cs n:05d>". These tags are not part of the subtitles themselves but indicate that the subtitle corresponds to the timestamp when the nth image is displayed, ensuring synchronization between the images and the subtitles.

"""
        return prompt

    def output_format(self) -> str:
        return """
**OUTPUT FORMAT**
The output format should be a JSON object, which is a list. 
Each element in the list is a natural language description containing exactly one user interaction and its purpose.
"""

    def get_prompts(self) -> str:
        prompts = self.outset() + self.images_input_format() + self.output_format()
        return prompts