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
The area highlighted with a red border in the screenshot indicates the region that has changed compared to the previous sampled frame. 
The red border is not part of the original video but is added later for marking purposes. 
The number in the top-left corner of the red border is a unique identifier for the border. 
The white text on a black background at the bottom center of the screenshot indicates the sequence number of the frame in the video.
2. Previous Key Frame:
The most recent key frame before the current frame. 
In this task, a key frame is defined as a frame where a user interaction has caused a significant change in the page, and the page has stabilized after the change. 
There must be at least one complete user interaction and the resulting page change between two key frames.
3. Previous Sampled Frame:
A screenshot of the page a few seconds before the current frame. 
There is a noticeable interface change between the current frame and the previous sampled frame.
4. Next Sampled Frame:
Usually, a screenshot of the page one second after the current frame. 
The chronological order of the frames is typically: Previous Key Frame -> Previous Sampled Frame -> Current Frame -> Next Sampled Frame.
5. Nearby Subtitles:
The explanatory subtitles from the video between the previous key frame and the next sampled frame. 
In the subtitles, tokens such as <Previous Key Frame>, <Previous Sampled Frame>, <Current Frame>, and <Next Sampled Frame> may appear. 
These tokens indicate the time when the corresponding frame appears in the video, and the narration happens to refer to this point.
6. Action to Be Completed:
The action that you need to judge. 
The goal is to determine whether the action has been completed at the time of the current frame.

At the beginning of the input, the following JSON data will be provided:
{
    "Current Frame": "Sequence number of the current frame",
    "Previous Key Frame": "Sequence number of the previous key frame",
    "Previous Sampled Frame": "Sequence number of the previous sampled frame",
    "Next Sampled Frame": "Sequence number of the next sampled frame, -1 means the next sampled frame does not exist",
    "Nearby Subtitles": "Subtitle content",
    "Action to Be Completed": "Action content"
}
Images of the four frames have been given to you at the beginning.

"""

    def reasoning_step(self) -> str:
        return """
**REASONING STEPS**
Please follow these steps to reason carefully.
1. First, determine whether the action to be completed was already completed before the previous key frame, and record the result.
2. If it was completed, skip the following steps and proceed directly to the output. 
If not, focus on the changes in the red-bordered area of the current frame compared to the previous sampled frame. 
Combine this with the corresponding subtitles to infer whether the changes were caused by user interaction.
3. If the changes were indeed caused by user interaction, mainly combine the next sampled frame to determine whether the "action to be completed" has been fully executed **before** (It's important) the current frame, meaning there are no significant page changes in the next sampled frame. 
It should not be halfway executed, with the next sampled frame still showing the action being performed or its impact on the page not yet finished.
If it is fully executed, record that the current frame is a key frame.
Because there may be cases where subtitles and images are not synchronized, you should not rely too heavily on subtitles for judgment, but instead primarily base your judgment on the images.
This is very important, please make sure to remember this step.

"""

    def output_format(self) -> str:
        return """
**OUTPUT FORMAT**
The output format must be and can only be a JSON object, which is like:
{
    "reasoning_step": "Your rigorous thought processes, in brief."
    "action_completed_early": 1/0,   # 1 means the action to be completed was completed before the previous key frame, 0 means it was not. If the value is 1, the following fields will be empty.
    "key_frame": 1/0,   # 1 means the current frame is determined to be a key frame, 0 means it is not.
    "action": "The content of the action to be completed."
}
"""

    def get_prompts(self) -> str:
        return self.outset() + self.input_format() + self.reasoning_step() + self.output_format()