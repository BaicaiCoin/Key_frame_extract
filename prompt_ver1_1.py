class Prompt:

    def __init__(self, interval):
        self.interval = interval

    def outset(self) -> str:
        return """
You are a data annotator, and you will receive video clips sourced from YouTube. Your task is to extract all the keyframes from the video clips.

**VIDEO DETAILS**
The video clips consist of tutorials on various computer applications, presented through screen recordings of the operations, accompanied by voiceovers or subtitles.

"""

    def key_frame_select_rules(self) -> str:
        return """
**KEYFRAMES SELECTION PRINCIPLE**
Selection Process:
1. Extract all "user interface interactions" from the video in detail, and store them in memory in chronological order.
2. Filter the interactions, removing redundant interactions and non-interaction events.
3. Combine the remaining interactions into "unit actions." Store them in memory.
4. Traverse all screenshots, and determine whether each screenshot is a keyframe based on the rules. If it is not a keyframe, delete it; if it is, save the token ID of the screenshot, as well as the operations executed between the previous keyframe and this keyframe.

User Interface Interactions:
A single operation of user interaction, such as:
-Click a button
-Enter a text
-Press a shortcut key
-And so on

Unit Action:
1. A unit action consists of one or a few user interface interactions.
2. If consecutive interactions do not require waiting for or monitoring the result of the previous interaction's effect on the page, and can simply be executed in sequence, they form a single unit action.
3. Examples and Explanation:
 -During a search, entering a keyword and clicking the search button is a unit action, as they can be executed sequentially. However, the next step, "clicking the correct entry," requires waiting for the search result list to load and checking it before knowing which item to select, so it cannot be combined into the same unit action.
 -A cascading dropdown cannot be a unit action, as the next level's options can only be decided which to be selected after the previous level's selection is made.

Non-interaction:
Any interface changes not caused by user interface interactions, most of which are due to post-production video edits, such as:
-Opening and closing animations
-Transition cuts
-And so on

Redundant Interaction:
Operations that are unrelated to the final goal, such as:
-Incorrect actions that are undone after execution
-Actions performed to help viewers understand the video, such as highlighting, zooming in or out, or scrolling the window
-And so on

Keyframe Selection Rules:
1. Delete frames that are not related to screen interface interactions.
2. Select the first screenshot where screen recording starts as the first keyframe.
3. For a screenshot, if a complete unit action has been executed after the previously selected keyframe, the page has stabilized, and the next screenshot clearly indicates the start of the next unit action, then this screenshot is selected as a keyframe.

"""

    def images_input_format(self) -> str:
        prompt = f"""
**INPUT FORMAT**
1. Due to the limitations of the input modality, the data consists of a combination of images and text, rather than the video itself.
2. The image-text combination starts with the [video start] tag and ends with the [video end] tag.
3. Images are screenshots taken every {self.interval} second from the video, while the text corresponds to the video's narration subtitles. You are required to infer the content of the video between these screenshots based on the time gaps. 
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
  "keyframe": "key frame screenshot ID",
  "goal": "Only for keyframes after the first one. A brief summary of what was done between the previous keyframe and this keyframe",
  "action": "Only for keyframes after the first one. A detailed description of the operations performed in the unit action corresponding to the keyframe"
}
"""

    def get_prompts(self) -> str:
        prompts = self.outset() + self.images_input_format() + self.key_frame_select_rules() + self.output_format()
        return prompts