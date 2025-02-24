class Prompt:

    def __init__(self, interval):
        self.interval = interval

    def key_frame_select_rules(self) -> str:
        return """
You are a data annotator, and you will receive video clips sourced from YouTube. Your task is to extract all the keyframes from the video clips.

**VIDEO DETAILS**
The video clips consist of tutorials on various computer applications, presented through screen recordings of the operations, accompanied by voiceovers or subtitles.

**KEYFRAMES SELECTION PRINCIPLE**
Selection Process:
1. Extract all "user interface interactions" from the video in sequence.
2. Filter the interactions, removing redundant or non-interaction events.
3. Combine the remaining interactions into "unit actions."
4. After each unit action is executed, wait for the page changes to stabilize, and then select the current frame as the keyframe.
5. Record the timestamp of the keyframe in the video, and note which operations were performed during this unit action. Write them down in the memory.

User interface interactions:
A single operation of user interaction, such as:
-Click a button
-Enter a text
-Press a shortcut key
-And so on

Unit Action:
Definition:
-Composed of one or a few interactions.
-Capable of causing changes on the page.
-After the execution of a unit action, the subsequent unit action must monitor the interface update to determine whether and how it can be executed. While there are no such restrictions between the operations within a unit action; they simply follow the order of execution.
Examples of unit actions include:
-Clicking a button to open a new page.
-Entering a keyword in the search box and clicking the search button.
-Modifying multiple configuration options within a settings interface.
-And so on.
To explain with the second example: entering a keyword and clicking the search button are simply executed in sequence, making them a single unit action. However, the next operation, selecting the target entry, requires observing the updated search result list to determine which item to choose.

Non-interaction:
Any interface changes not caused by user interface interactions, most of which are due to post-production video edits, such as:
-Opening and closing animations
-Transition cuts
-And so on

Redundant interaction:
Operations that are unrelated to the final goal, such as:
-Incorrect actions that are undone after execution
-Actions performed to help viewers understand the video, such as highlighting, zooming in or out, or scrolling the window
-And so on



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

    def video_input_format(self) -> str:
        return """

"""

    def output_format(self) -> str:
        return """
**OUTPUT FORMAT**
The output format should always be a JSON object, which is a list containing the following fields:
{
  "keyframe": "keyframe image number",
  "goal": "A brief summary of what was done between the previous keyframe and this keyframe",
  "action": "A detailed description of the operations performed in the unit action corresponding to the keyframe"
}
"""

    def get_prompts(self) -> str:
        prompts = self.key_frame_select_rules() + self.images_input_format() + self.output_format()
        return prompts