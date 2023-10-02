import json
import pickle
import random
import re
import textwrap
from functools import partial
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset, DatasetDict
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from utils.prompt_utils import make_observation_prompt
from utils.vector_utils import (
    EgoFieldRandom,
    PedestrianFieldRandom,
    RouteFieldRandom,
    VehicleFieldRandom,
    get_tl_state,
)

INSTRUCTION = "You are a certified professional driving instructor and please tell me step by step how to drive a car based on the input scenario."
ACTION_Q = [
    "How are you going to drive in this situation?",
    "What actions are you taking?",
    "Why are you taking these specific driving actions?",
    "How are you navigating this situation?",
    "What actions will be taken in this situation?",
    "How are you driving in this situation?",
    "What are your current actions?",
    "What are your planned actions for this situation?",
    "How are you managing the car?",
    "What are your actions?",
]
DEFAULT_EVAL_ITEMS = ["caption", "action"]


def get_val_data(
    val_data_path,
    tokenizer,
    val_set_size=1,
    eval_items=None,
    add_input_prompt=False,
    legacy_parser=False,
):
    return _val_data_from_val_dataset(
        val_dataset=_load_val_dataset(
            val_data_path,
            val_set_size,
            eval_items=eval_items,
            add_input_prompt=add_input_prompt,
            legacy_parser=legacy_parser,
        ),
        tokenizer=tokenizer,
    )


def _val_data_from_val_dataset(val_dataset, tokenizer):
    return val_dataset["test"].map(
        partial(generate_and_tokenize_prompt, tokenizer, user_input_ids=True),
        remove_columns=[],
        num_proc=8,
    )


def get_train_val_data(
    data_path,
    tokenizer,
    val_data_path=None,
    val_set_size=1,
    augment_times=1,
    load_pre_prompt_dataset=False,
    vqa=False,
    add_input_prompt=False,
    eval_only=False,
    eval_items=None,
):
    if eval_only:
        return None, get_val_data(
            val_data_path, tokenizer, val_set_size, eval_items=eval_items
        )

    assert data_path.endswith(".pkl"), "Only support pkl data format"
    if vqa:
        train_dataset = _load_vqa_train_dataset(
            data_path, add_input_prompt=add_input_prompt
        )
    elif load_pre_prompt_dataset:
        train_dataset = _load_pre_prompt_dataset(data_path, augment_times)
    else:
        train_dataset = _load_vector_pkl_dataset(data_path, augment_times)

    if val_set_size > 0:
        if val_data_path is not None:
            val_dataset = _load_val_dataset(
                val_data_path, val_set_size, eval_items=eval_items
            )
        else:
            train_val = train_dataset["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_dataset = train_val
            val_dataset = train_val

        train_data = train_dataset["train"].shuffle(seed=42)
        train_data = train_data.map(
            partial(generate_and_tokenize_prompt, tokenizer),
            remove_columns=[],
            num_proc=8,
        )

        val_data = _val_data_from_val_dataset(
            val_dataset=val_dataset, tokenizer=tokenizer
        )
    else:
        train_data = train_dataset["train"].shuffle(seed=42)
        train_data = train_data.map(
            generate_and_tokenize_prompt, remove_columns=[], num_proc=8
        )
        val_data = None
    return train_data, val_data


def generate_and_tokenize_prompt(tokenizer, data_point, user_input_ids=False):
    full_prompt = generate_prompt(data_point)

    tokenized_full_prompt = tokenize(tokenizer, full_prompt)

    user_prompt = generate_prompt({**data_point, "output": ""})
    tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=False)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt[
        "labels"
    ][
        user_prompt_len:
    ]  # could be sped up, probably

    if user_input_ids:
        tokenized_user_prompt = tokenize(
            tokenizer,
            user_prompt,
            padding="max_length",
            cutoff_len=86,
            add_eos_token=False,
        )
        tokenized_full_prompt["user_input_ids"] = tokenized_user_prompt["input_ids"]
        tokenized_full_prompt["user_attention_mask"] = tokenized_user_prompt[
            "attention_mask"
        ]
    tokenized_full_prompt["route_descriptors"] = data_point["route_descriptors"]
    tokenized_full_prompt["vehicle_descriptors"] = data_point["vehicle_descriptors"]
    tokenized_full_prompt["pedestrian_descriptors"] = data_point[
        "pedestrian_descriptors"
    ]
    tokenized_full_prompt["ego_vehicle_descriptor"] = data_point[
        "ego_vehicle_descriptor"
    ]

    return tokenized_full_prompt


def tokenize(tokenizer, prompt, cutoff_len=512, padding=False, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=padding,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()

    return result


def get_traffic_light_labels(route_descriptors):
    tl_state, tl_distance = get_tl_state(route_descriptors)
    tl_prompt = ""
    if tl_state is None:
        tl_prompt += f"There is no traffic lights.\n"
    else:
        tl_prompt += f"There is a traffic light and it is {tl_state}. It is {tl_distance:.2f}m ahead.\n"
    return tl_prompt


def generate_prompt(data_point):
    if "instruction" not in data_point:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{INSTRUCTION}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
        # sorry about the formatting disaster gotta move fast
    if "input" in data_point:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""

    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""


def parse_vqa_response_content(response_content):
    vqa_data = re.findall(r"\{.*?\}", response_content)
    for json_string in vqa_data:
        json_string = json_string.replace("\t", " ").replace("\n", " ")
        json_string = json_string.replace(",}", "}")
        json_string = json_string.replace("`", '"').replace('''', "''', '", "')
        try:
            json_dict = json.loads(json_string)
            assert "question" in json_dict
            assert "answer" in json_dict
        except Exception:
            continue
        yield json_dict


def parse_vqa_response_content_legacy(response_content):
    for line in response_content.split("\n"):
        try:
            json_dict = json.loads(line)
            assert "question" in json_dict
            assert "answer" in json_dict
            yield json_dict
        except Exception:
            continue


def _load_vqa_pickle_dataset(
    data_path,
    add_input_prompt=False,
    dataset_items=None,
    max_size=None,
    legacy_parser=False,
):
    if legacy_parser:
        # used to prepare test datasets for the paper
        parser = parse_vqa_response_content_legacy
    else:
        parser = parse_vqa_response_content

    if dataset_items is None:
        dataset_items = ["vqa", "caption", "action"]
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    data_dict: Dict[str, List[Any]] = {
        "frame_num": [],
        "input": [],
        "instruction": [],
        "output": [],
        "route_descriptors": [],
        "vehicle_descriptors": [],
        "pedestrian_descriptors": [],
        "ego_vehicle_descriptor": [],
    }
    if add_input_prompt:
        data_dict["input_prompt"] = []
    for d in data:
        # VQA
        if "vqa" in dataset_items:
            obs_dict = d["observation"]
            for json_dict in parser(d["response_content"]):
                data_dict["frame_num"].append(d["frame_num"])
                data_dict["input"].append("")
                data_dict["instruction"].append(json_dict["question"])
                data_dict["output"].append(json_dict["answer"])
                _append_descriptors(data_dict, obs_dict)
                if add_input_prompt:
                    data_dict["input_prompt"].append(d["input_prompt"])
        # Captioning
        if "caption" in dataset_items:
            data_dict["frame_num"].append(d["frame_num"])
            data_dict["input"].append("")
            data_dict["instruction"].append(INSTRUCTION)
            data_dict["output"].append(make_observation_prompt(d["observation"]))
            _append_descriptors(data_dict, d["observation"])
            if add_input_prompt:
                data_dict["input_prompt"].append(d["input_prompt"])

        # Action
        if "action" in dataset_items:
            data_dict["frame_num"].append(d["frame_num"])
            data_dict["input"].append("")
            data_dict["instruction"].append(
                ACTION_Q[len(data_dict["instruction"]) % len(ACTION_Q)]
            )
            data_dict["output"].append("\n".join(d["input_prompt"].split("\n")[-4:]))
            _append_descriptors(data_dict, d["observation"])
            if add_input_prompt:
                data_dict["input_prompt"].append(d["input_prompt"])

        if max_size is not None and len(data_dict["frame_num"]) >= max_size:
            break

    return Dataset.from_dict(data_dict)


def _load_vqa_train_dataset(data_path, add_input_prompt=False):
    training_data = _load_vqa_pickle_dataset(
        data_path, add_input_prompt=add_input_prompt
    )
    dataset = DatasetDict(train=training_data)
    return dataset


def _load_val_dataset(
    data_path,
    val_set_size,
    eval_items=None,
    add_input_prompt=False,
    legacy_parser=False,
) -> DatasetDict:
    if eval_items is None:
        eval_items = DEFAULT_EVAL_ITEMS
    val_data = _load_vqa_pickle_dataset(
        data_path,
        add_input_prompt=add_input_prompt,
        dataset_items=eval_items,
        max_size=val_set_size,
        legacy_parser=legacy_parser,
    )
    dataset = DatasetDict(test=val_data)
    return dataset


def _append_descriptors(data_dict, obs_dict):
    for k in [
        "route_descriptors",
        "vehicle_descriptors",
        "pedestrian_descriptors",
        "ego_vehicle_descriptor",
    ]:
        data_dict[k].append(obs_dict[k])


def _get_random_obs(obs_dict):
    # Zero out the input
    for k, v in obs_dict.items():
        if k in [
            "route_descriptors",
            "vehicle_descriptors",
            "pedestrian_descriptors",
            "ego_vehicle_descriptor",
        ]:
            obs_dict[k] = v * 0.0

    # RouteField
    for i in range(len(obs_dict["route_descriptors"])):
        RouteFieldRandom.randomize(obs_dict["route_descriptors"][i], has_tl=False)
    has_tl = random.uniform(0, 1) <= 0.75
    if has_tl:
        tl_slot = random.randint(0, len(obs_dict["route_descriptors"]) - 1)
        RouteFieldRandom.randomize(obs_dict["route_descriptors"][tl_slot], has_tl=True)

    # VehicleField
    for i in range(len(obs_dict["vehicle_descriptors"])):
        if random.uniform(0, 1) <= 0.25:
            break
        VehicleFieldRandom.randomize(obs_dict["vehicle_descriptors"][i])

    # PedestrianField
    for i in range(len(obs_dict["pedestrian_descriptors"])):
        if random.uniform(0, 1) <= 0.3:
            break
        PedestrianFieldRandom.randomize(obs_dict["pedestrian_descriptors"][i])

    # EgoField
    EgoFieldRandom.randomize(obs_dict["ego_vehicle_descriptor"])

    return obs_dict


def _load_pre_prompt_dataset(data_path, augment_times):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    data = data["observations"]
    data_dict: Dict[str, List[Any]] = {
        "input": [],
        "output": [],
        "route_descriptors": [],
        "vehicle_descriptors": [],
        "pedestrian_descriptors": [],
        "ego_vehicle_descriptor": [],
    }
    # Add synthetic data
    for _ in tqdm(range(augment_times), desc="Augment times"):
        for d in tqdm(data, desc="Processing data", leave=False):
            data_dict["input"].append("")
            data_dict["output"].append(make_observation_prompt(d))
            _append_descriptors(data_dict, _get_random_obs(d))
    # Add real data
    for d in data:
        data_dict["input"].append("")
        data_dict["output"].append(make_observation_prompt(d))
        _append_descriptors(data_dict, d)
    training_data = Dataset.from_dict(data_dict)
    dataset = DatasetDict(train=training_data)
    return dataset


def _load_vector_pkl_dataset(data_path, augment_times):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    data_dict: Dict[str, List[Any]] = {
        "frame_num": [],
        "input": [],
        "output": [],
        "route_descriptors": [],
        "vehicle_descriptors": [],
        "pedestrian_descriptors": [],
        "ego_vehicle_descriptor": [],
    }
    # Add synthetic data
    for _ in tqdm(range(augment_times), desc="Augment times"):
        for d in tqdm(data, desc="Processing data", leave=False):
            data_dict["frame_num"].append(-1)
            data_dict["input"].append("")
            data_dict["output"].append(make_observation_prompt(d["observation"]))
            _append_descriptors(data_dict, _get_random_obs(d["observation"]))

    # Add real data
    for d in data:
        data_dict["frame_num"].append(d["frame_num"])
        data_dict["input"].append("")
        data_dict["output"].append(make_observation_prompt(d["observation"]))
        _append_descriptors(data_dict, d["observation"])

    training_data = Dataset.from_dict(data_dict)
    dataset = DatasetDict(train=training_data)
    return dataset


def log_txt_as_img(wh, xc, size=10):
    # wh: a tuple of (width, height)
    # xc: a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("font/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))

        # Split the input string using \n
        lines = xc[bi].split("\n")

        # Wrap each line using textwrap.wrap() and draw the lines with proper vertical offset
        y_offset = 0
        for line in lines:
            wrapped_line = textwrap.wrap(line, width=nc)
            for wrapped_segment in wrapped_line:
                draw.text((0, y_offset), wrapped_segment, fill="black", font=font)
                y_offset += font.getsize(wrapped_segment)[1]

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = np.transpose(txts, (0, 2, 3, 1))
    return txts


def get_control_longitudinal(text):
    accelerator = re.search(r"Accelerator pedal (\d+)%", text)
    brake = re.search(r"Brake pedal (\d+)%", text)
    accelerator_value = int(accelerator.group(1)) / 100 if accelerator else None
    brake_value = int(brake.group(1)) / 100 if brake else None
    if accelerator_value is None or brake_value is None:
        return None
    x = accelerator_value - brake_value
    control_longitudinal = (x + 1.0) / 2.0
    return control_longitudinal


def get_control_lateral(text):
    match = re.search(r"(\d+)% to the (right|left)\.", text, re.IGNORECASE)
    if match:
        percentage, direction = match.groups()
        value = int(percentage) / 100.0
        if direction.lower() == "right":
            value *= -1
        return value
    return None


def eval_tl(all_pred, all_label):
    no_tl_correct = 0
    no_pred_contains_tl = 0
    for pred, label in zip(all_pred, all_label):
        if "no traffic lights" in label:
            if "no traffic lights" in pred:
                no_tl_correct += 1
        elif "red+yellow" in label:
            if "red+yellow" in pred:
                no_tl_correct += 1
        elif " red" in label:
            if " red" in pred:
                no_tl_correct += 1
        elif " green" in label:
            if " green" in pred:
                no_tl_correct += 1
        elif " yellow" in label:
            if " yellow" in pred:
                no_tl_correct += 1
        else:
            continue
        no_pred_contains_tl += 1

    if no_pred_contains_tl == 0:
        return None

    accuracy = no_tl_correct / no_pred_contains_tl
    return accuracy


def get_eval_distance_errors(all_pred, all_label, pattern):
    def extract_number(text):
        # Find the first match in the string
        match = re.search(pattern, text)
        # Return the float value if a match is found, otherwise return None
        return float(match.group(1)) if match else None

    distance_errors = []
    for pred, label in zip(all_pred, all_label):
        label_distance = extract_number(label)
        pred_distance = extract_number(pred)
        if label_distance is not None and pred_distance is not None:
            distance_errors.append(abs(label_distance - pred_distance))
        elif label_distance is not None and pred_distance is None:
            distance_errors.append(abs(label_distance))

    return distance_errors


def eval_action(all_pred, all_label):
    accumulated_error_lon = 0.0
    accumulated_error_lat = 0.0
    average_error_lon = 0.0
    average_error_lat = 0.0
    total_number = 0

    for pred, label in zip(all_pred, all_label):
        pred_control_longitudinal = get_control_longitudinal(pred)
        label_control_longitudinal = get_control_longitudinal(label)
        pred_control_lateral = get_control_lateral(pred)
        label_control_lateral = get_control_lateral(label)
        if (
            pred_control_longitudinal is None
            or label_control_longitudinal is None
            or pred_control_lateral is None
            or label_control_lateral is None
        ):
            continue
        control_error_lon = abs(pred_control_longitudinal - label_control_longitudinal)
        control_error_lat = abs(pred_control_lateral - label_control_lateral)
        accumulated_error_lon += control_error_lon
        accumulated_error_lat += control_error_lat
        total_number += 1
    if total_number > 0:
        average_error_lon = accumulated_error_lon / total_number
        average_error_lat = accumulated_error_lat / total_number
    return average_error_lon, average_error_lat


def decode_generation_seqeunces(tokenizer, token_sequences):
    token_sequences = np.where(
        token_sequences != -100, token_sequences, tokenizer.pad_token_id
    )
    return tokenizer.batch_decode(
        token_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
