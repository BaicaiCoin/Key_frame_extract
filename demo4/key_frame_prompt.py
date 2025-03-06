class Prompt:
    def __init__(self):
        pass

    def outset(self) -> str:
        return """
You are a data annotator.
You will receive an image, which is a screenshot of a specific frame extracted from a video, with highlighted areas.
Additionally, you will be provided with subtitles related to the video and several other screenshots of nearby frames. 
Your task is to analyze these contents to determine the type of change within the highlighted area and decide whether this frame qualifies as a key frame.
Below is a detailed description of the task.

**VIDEO DETAILS**
The content of the video is a tutorial on the operation of various computer applications. 
These videos are from YouTube and primarily consist of screen recordings of computer operations.

"""

    def input_format(self) -> str:
        return """
**INPUT FORMAT**
The input information consists of the following six parts:

1. Current Frame:
The screenshot that you need to judge. 
The area highlighted with a red border in the screenshot indicates the region that has changed compared to the nearest previous sampled frame. 
The red border is not part of the original video but is added later for marking purposes. 
The purpose of the red border is to draw your attention to the pixel changes in this area. So reasoning must not include thoughts such as "the action has been executed because it is highlighted by the red border."
The number in the top-left corner of the red border is a unique identifier for the border. 

2. Previous Key Frame:
The most recent key frame before the current frame. 
In this task, a key frame is defined as a frame where a user interaction has caused a significant change in the page, and the page has stabilized after the change. 
There must be at least one complete user interaction and the resulting page change between two key frames.

3. Previous Sampled Frames:
Screenshots of the page taken before the current frame, which may include 0 to 2 sampled frames.
The time intervals between previous sampled frames and between previous sampled frame and the current frame are not fixed; they are selected based on page pixel changes. 

4. Next Sampled Frames:
Screenshots of the page taken after the current frame, which may include 0 to 2 sampled frames.
The intervals between next sampled frames and between the current frame and next sampled frame are fixed, usually 1 second.

The chronological order of the frames is typically: Previous Key Frame -> Previous Sampled Frames -> Current Frame -> Next Sampled Frames.
All frames have a sequence number displayed at the bottom center of the screenshot, in white text on a black background, indicating the frame's order in the video.

5. Nearby Subtitles:
The explanatory subtitles from the video between the previous key frame and the next sampled frame. 
In the subtitles, tokens such as <Previous Key Frame>, <Previous Sampled Frame 0>, <Previous Sampled Frame 1>, <Current Frame>, <Next Sampled Frame 0>, and <Next Sampled Frame 1> may appear. 
These tokens indicate the time when the corresponding frame appears in the video, and the narration happens to refer to this point.

6. Action to Be Completed:
The action that you need to judge.

The following JSON data will be provided:
{
    "Current Frame": "Sequence number of the current frame",
    "Previous Key Frame": "Sequence number of the previous key frame",
    "Previous Sampled Frames": "List of sequence number of the previous sampled frames",
    "Next Sampled Frames": "List of sequence number of the next sampled frames",
    "Nearby Subtitles": "Subtitle content",
    "Action to Be Completed": "Action content"
}
The frames' images are provided to you in the chronological order of the video.

"""

    def reasoning_step(self) -> str:
        return """
**REASONING STEPS**
Please follow these steps to reason carefully.
1. First, you need to determine whether the current frame is a screen recording of computer operations. If not, skip the following steps and directly mark the current frame as not a keyframe.
2. Then, compare the changes within the red box in the current frame with the previous sampled frame to infer whether the changes were caused by user interaction.
3. If the changes were indeed caused by user interaction, mainly combine the next sampled frame to determine whether: **the action has been fully executed, and the page will not change again before the next action is performed.**(It's important, please remember it)
Note that the result of the action must also appear before the current frame for it to be defined as "fully executed".
The difference between the current frame and the previous sampled frame does not indicate that the action has been fully executed. Please do not make such a judgment.
If it is fully executed, record that the current frame is a key frame.
Because there may be cases where subtitles and images are not synchronized, you should not rely too heavily on subtitles for judgment, but instead primarily base your judgment on the images.
This is very important, please make sure to remember this step.
For example:
-When clicking a button to open a new page, you cannot judge the current frame as a key frame based on the content displayed when the mouse hovers over the button. Instead, you should select the frame after the new page appears as the key frame.
-When entering a keyword in a search box and the search results change instantly, you should select the frame after the keyword has been fully entered as the key frame.

"""

    def output_format(self) -> str:
        return """
**OUTPUT FORMAT**
The output format must be and can only be a JSON object, which is like:
{
    "reasoning_step": "Your rigorous thought processes, in brief."
    "key_frame": 1/0,   # 1 means the current frame is determined to be a key frame, 0 means it is not.
    "action": "The content of the action to be completed."
}
"""

    def get_prompts(self) -> str:
        return self.outset() + self.input_format() + self.reasoning_step() + self.output_format()
    
    def verify_result(self, result_frame, action) -> str:
        prompt = f"""
You are a data validator, and your task is to verify whether the results produced by another LLM are correct.

**TASK DETAILS**
You will receive several screenshot frames from a video clip about computer application tutorials. The sequence is arranged in chronological order based on the video's timeline.
Each frame has a number displayed at the bottom center, indicating its position in the video clip.
You will then be provided with the "selected frame" number and a description of a user interaction action. The selected frame will be one of the frames provided.
Your task is to infer the content of this video segment and determine whether the selected frame represents an "intermediate state case" of the given action execution, mainly focusing on reviewing the content of the subsequent frames
The "intermediate state cases" are listed below, and only these cases apply. Please do not infer or assume any other cases.
If it matches any of the listed cases, output "True" at the end; if not, output "False".

Cases of intermediate states:
-A hover effect is triggered when the mouse is over a button or other interactive elements, but no click has been made.
-Only part of the text has been entered, but the input is not yet complete.
-The current page is blank, which means most elements have not been loaded.

**OUTPUT FORMAT**
Your output should be a JSON object in the following format:
{{
  "result": Boolean, 
  "reason": "The brief reason for your inference."
}}

[Input data Start]
selected frame: {result_frame}
user interaction action: {action}
[Input Data End]
"""
        return prompt