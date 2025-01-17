import json
import os
import re
from typing import Dict, Any, Union

from .base import BaseEvalDataset


TASKS = {
    "reasoning_types": [
        "count",
        "direction",
        "rotation",
        "shape&trend",
        "velocity&frequency",
        "visual_cues"
    ],
    "demonstration_types": [
        "human",
        "object",
        "simulated"
    ]
}


class TOMATODataset(BaseEvalDataset):

    BENCHMARK_TYPE: str = "mcqa"

    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}
        idx = 0

        for task_name in TASKS["reasoning_types"]:
            json_file = os.path.join(data_root, "data", task_name + '.json')
            video_folder = os.path.join(data_root, "videos")

            with open(json_file, 'r') as f:
                task_data_list = json.load(f)

            for key in task_data_list:
                data = task_data_list[key]

                answer  = str(data["answer"])
                options = data["options"]

                option_letters = []
                for option_idx, option in enumerate(options): 
                    option_letters.append(f"{chr(ord('A') + option_idx)}")
                    if option == answer:
                        answer_idx = option_idx

                data_dict[idx] = {
                    # required fields for data loading
                    "video_path": os.path.join(video_folder, data['demonstration_type'], data["key"] + ".mp4"),
                    "start_time": None,
                    "end_time": None,
                    # required fields for evaluation
                    "task_type": task_name,
                    "ground_truth": answer_idx,
                    # custom fields for instruction generation and post processing
                    "question": data["question"],
                    "options": options,
                    "option_letters": option_letters,
                }
                idx += 1

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video: Any) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        option_letters = meta_data["option_letters"]
        options = meta_data["options"]

        option_string = ""
        for option_idx, (letter, option) in enumerate(zip(option_letters, options)):
            option_string += f"({letter}) {option}\n"
        instruction = f"Question: {question}\nOptions:\n{option_string}Answer with the option\'s letter from the given choices directly and only give the best option."

        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        meta_data = self.data_dict[data_id]
        options = meta_data["options"]
        option_letters = meta_data["option_letters"]

        response = response.replace('answer', '')
        response = response.replace('Answer', '')
        pred_answer = re.findall(f'[\(,\ ]*[{option_letters[0]}-{option_letters[-1]}][\),\ ]*', response)

        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                opt = opt.strip()
                opt = opt.strip('.')
                # Arabic numerals -> English words
                if opt.lower() in output.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = option_letters.index(pred_answer)
            find_flag = True

        assert find_flag, f"Cannot find the answer in the options: {response}"
        return pred_idx
