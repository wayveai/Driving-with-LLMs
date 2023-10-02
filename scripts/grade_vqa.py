import copy
import csv
import json
import pickle
import re
from multiprocessing import pool as mpp
from pathlib import Path
from typing import Optional

import click
import numpy as np
import openai
import random
from retry import retry
from tqdm import tqdm

SYSTEM_PROMPT = f"""You are a certified professional driving instructor in London, UK.
Your job is to teach students to drive and to grade their answers to a standardised driving test consisting of questions related to driving in an urban environment.
Your colleagues describe you as careful, charismatic, and smart. You always care about correct judgement of students.

## London driving
In London, we drive on the left side of the road according to the UK driving rules.
A red and yellow signal on a traffic light is typically used as a warning signal to indicate that the light is about to turn green.
As we drive, students observe multiple objects such as vehicles, pedestrians, and traffic lights around us.
If a car is driving in an opposite direction and is to the right of us, it is driving on an opposite lane.

## Units
For each object, the student must be aware of their direction and distance from our car. In the standardised test, we measure distances in meters and angles in degrees.
Positive distances mean the object is in front. Negative distances mean it's behind us.
An angle of 0 indicates an object is straight-ahead. An angle of 90 indicates the object is to the right, and -90 indicates it's on the left.
This means negative angles indicate the object is to the left of us, and positive angles that it's to the right of us.
An angle >= 90 degrees is considered to be a sharp angle: e.g 90 degrees is a sharp right, -90 degrees is a sharp left.

## Student test
Students should drive in a defensive way and they should pay varying levels of attention to different objects.
In the test we measure attention as a percentage from 0% to 100%, depending on how they might be a hazard that may cause me to change speed, direction, stop, or even cause harm to myself.
In your grading, it's REALLY important to check answers for their factually correctness. Even if the student gives a reasonable sounding answer, it might be factually incorrect.
Always think critically."""


def parse_question_answers(response):
    vqa_data = re.findall(r"\{.*?\}", response)
    for json_string in vqa_data:
        json_string = json_string.replace("\t", " ").replace("\n", " ")
        json_string = json_string.replace(",}", "}")
        json_string = json_string.replace("`", '"').replace('''', "''', '", "')
        try:
            json_dict = json.loads(json_string)
            if not json_dict["question"] or not json_dict["answer"]:
                continue
        except Exception:
            continue
        yield json_dict


class PoolWithTQDM(mpp.Pool):
    def istarmap(self, func, iterable, chunksize=1):
        """starmap-version of imap - allows use of tqdm for progress bar with pool"""
        self._check_running()  # type: ignore
        if chunksize < 1:
            raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

        task_batches = self._get_tasks(func, iterable, chunksize)  # type: ignore
        result = mpp.IMapIterator(self)  # type: ignore
        self._taskqueue.put(  # type: ignore
            (
                self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),  # type: ignore
                result._set_length,  # type: ignore
            )
        )
        return (item for chunk in result for item in chunk)

    def __reduce__(self):
        raise NotImplementedError("Pool objects cannot be passed between processes or pickled.")


@retry(tries=16, delay=2, backoff=2, max_delay=10, jitter=(0, 5))
def grade(observation_prompt, question_answer_pairs, pred):
    input_prompt = f"""You are now given a description of the student's observation. This observation includes their attention to objects as well as position and direction. Your task is to grade the answers.

### Scoring rules:
- Your scores for each answer should be between 0 (worst) and 10 (best).
- If the answer is totally wrong or the numbers are way off, the score should be 0.
- Give intermediate scores for partially correct answers, or numbers that are close. Only give a score of 10 for flawless answers.
- If the question is unrelated to driving, the only thing you need to check is if the student acknowledges that the question is unrelated to driving.
- Don't assess the student's observation nor their attention. They are unable to control it. Focus on their answers instead.
- Think critically and carefully. Most importantly check the answer for factual correctness.


### Grading process

For each of the {len(question_answer_pairs)} questions, provide a one line assessment of the student's answer in the format:

```
n. <short one sentence assessment>. Score: <score 0-10>.
```

A few examples of what a good assessment might look like:
1. Correctly identified the the red traffic light. Score: 10.
2. No mention of pedestrian crossing street. Score: 2.
3. Acknowledged question unrelated to driving and attempted to answer. Score: 10.
4. Doesn't answer the question but has all information necessary. Score: 0.
5. Doesn't stop for the pedestrian ahead. Score: 0.
6. Correctly identified the attention level to the car. Score: 10.
7. Incorrectly stated there are no vehicles around even though there is one. Score: 0.
8. Unable to answer question given information available. Score: 10.
9. Give 13.12 mph or m or percentage for the correct answer of 13.0. Score: 8.
10. Give 14.12 mph or m or percentage for the correct answer of 13.0. Score: 6.


### Student's observation:
{observation_prompt}


### Student's questionnaire:
"""
    for i, qa in enumerate(question_answer_pairs):
        if pred is not None:
            # Overwrite the answer with the prediction
            prediction = pred[i]
            question_answer_pairs[i]['answer'] = prediction
        else:
            prediction = qa['answer']
        input_prompt += f"""Question {i+1}: {qa['question']}
Answer: {prediction}

"""

    input_prompt += f"""


### Your assessment:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_prompt},
        ],
        top_p=0.8,
        max_tokens=1000,
    )

    assessment = response['choices'][0]['message']['content']
    pattern = re.compile(r'(\d+)\.\s*(.*)\s*Score:\s*(\d+)')

    graded_qa_pairs = copy.deepcopy(question_answer_pairs)
    for line in assessment.split('\n'):
        match = pattern.match(line)
        if match is None:
            continue
        question_index = int(match.group(1)) - 1
        explanation = match.group(2).strip()
        score = int(match.group(3))
        if not 0 <= question_index < len(question_answer_pairs):
            continue
        graded_qa_pairs[question_index]['score'] = score
        graded_qa_pairs[question_index]['score_explanation'] = explanation

    return graded_qa_pairs


def process_frame(frame, verbose=False):
    observation_prompt = frame['input_prompt']
    if verbose:
        print("========= =========")
        print("Observation: ")
        print(observation_prompt)
    question_answer_pairs = list(parse_question_answers(frame['response_content']))
    grade_result = grade(observation_prompt, question_answer_pairs, frame.get("pred"))

    if verbose:
        for qa_pair in grade_result:
            print("========= =========")
            print(f"Question: {qa_pair['question']}")
            print(f"Answer: {qa_pair['answer']}")
            if 'score' in qa_pair:
                print(f"Score: {qa_pair['score']}")
                print(f"Explanation: {qa_pair['score_explanation']}")
            else:
                print("Score: N/A")
                print("Explanation: N/A")

    frame['response_content'] = '\n'.join([json.dumps(qa) for qa in grade_result])
    return frame

def maybe_filter_result(data, result):
    data_total_entries = 0
    for d in data:
        data_total_entries += len(list(parse_question_answers(d["response_content"])))
    if len(result) != data_total_entries:
        result = filter_vqa_result(data, result)
    assert data_total_entries == len(result), f"len(data)={data_total_entries} != len(result)={len(result)}, consider run filter_vqa_result(data, result)"
    return result

def filter_vqa_result(data, result):
    dataset_items=["vqa", "caption", "action"]
    data_dict= {
        "frame_num": [],
        "input": [],
        "instruction": [],
        "output": [],
        "route_descriptors": [],
        "vehicle_descriptors": [],
        "pedestrian_descriptors": [],
        "ego_vehicle_descriptor": [],
    }
    for d in data:
        # VQA
        if "vqa" in dataset_items:
            obs_dict = d["observation"]
            for json_dict in parse_question_answers(d["response_content"]):
                data_dict["frame_num"].append(d["frame_num"])
                data_dict["input"].append("")
                data_dict["instruction"].append(json_dict["question"])
                data_dict["output"].append(json_dict["answer"])

        # Captioning
        if "caption" in dataset_items:
            data_dict["frame_num"].append(d["frame_num"])
            data_dict["input"].append("")
            data_dict["instruction"].append("")
            data_dict["output"].append("")


        # Action
        if "action" in dataset_items:
            data_dict["frame_num"].append(d["frame_num"])
            data_dict["input"].append("")
            data_dict["instruction"].append("")
            data_dict["output"].append("\n".join(d["input_prompt"].split("\n")[-4:]))
    filtered_result = []
    for d,r in zip(data_dict["instruction"], result):
        if d != "":
            filtered_result.append(r)
    return filtered_result


def load_result(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract the 'data' list from the loaded JSON data
    data_list = data["data"]

    # Initialize an empty list to store the dictionaries
    result = []

    # Iterate over each sublist in data_list
    for sublist in data_list:
        # Create a dictionary for each sublist and append it to the result list
        result.append({
            "pred": sublist[0].split("\n")[-1],
            "label": sublist[1]
        })

    return result


def get_avg_score(results):
    all_scores = []
    for res in results:
        resp = res["response_content"]
        # Splitting the string into separate lines (JSON objects)
        json_lines = resp.strip().split("\n")
        scores = []

        for line in json_lines:
            try:
                loaded_json = json.loads(line)
                if "score" in loaded_json:
                    scores.append(loaded_json["score"])
            except json.JSONDecodeError:
                continue  # Skip the line if JSON decoding fails
        if scores:
            all_scores.extend(scores)
    # print(all_scores)
    print("avg score:", np.mean(all_scores))


def save_list_of_dicts_to_csv(list_of_dicts, file_path, ignored_columns=None):
    if ignored_columns is None:
        ignored_columns = []

    # Ensure the list is not empty
    if len(list_of_dicts) == 0:
        return

    # Get keys but exclude those in ignored_columns
    keys = [k for k in list_of_dicts[0].keys() if k not in ignored_columns]

    with open(file_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        
        # Only include rows with keys that we're interested in
        for dictionary in list_of_dicts:
            filtered_dict = {k: dictionary[k] for k in keys}
            dict_writer.writerow(filtered_dict)
        

@click.command()
@click.option(
    '-i',
    'evaluation_data_path',
    type=click.Path(path_type=Path),
    required=True,
    help='Path to evaluation result file. This should have the same format as the dataset produced by make_prompt_vqa.py.',
)
@click.option(
    '-r',
    'evaluation_result_path',
    type=click.Path(path_type=Path),
    help='Path to evaluation result file. This should have the same format as the dataset produced by make_prompt_vqa.py.',
)
@click.option(
    '-o',
    'output_path',
    type=click.Path(path_type=Path, file_okay=True, exists=False),
    help='Path to output file. This should have the same format as the dataset produced by make_prompt_vqa.py with an extra .',
)
@click.option(
    '-l', '--limit', 'limit_n', type=int, default=None, help='Limit the number of examples to grade to the first n.'
)
@click.option('-d', '--debug', is_flag=True, default=False, help='Debug mode.')
@click.option('-w', 'num_workers', type=int, default=1, help='Number of workers to use.')
@click.option("--openai_api", required=True, type=str)
@click.option('-s', 'shuffle', is_flag=True, default=False, help='[DANGEROUS] Shuffle the label to create wrong answers to questions')
@click.option('-k', '--idk', is_flag=True, default=False, help='[DANGEROUS] Overwrite the label with "I dont know" to create wrong answers to questions')
def grade_vqa(
    evaluation_data_path: Path,
    evaluation_result_path: Path,
    output_path: Optional[Path],
    limit_n: Optional[int],
    debug: bool = False,
    num_workers: int = 1,
    openai_api: str = None,
    shuffle: bool = False,
    idk: bool = False,
):
    """
    Evaluate the outputs of a Vector QA model using the OpenAI api.

    Example outputs:
        ~/isim_obs_8k_vqa.pkl
        /mnt/remote/data/users/sinavski/vec_llm/isim_obs_8k_vqa.pkl
    """
    openai.api_key = openai_api

    assert evaluation_data_path.exists(), f"evaluation_data_path={evaluation_data_path} does not exist."

    with open(evaluation_data_path, "rb") as f:
        data = pickle.load(f)
    result = load_result(evaluation_result_path) if evaluation_result_path else None
    if result:
        result = maybe_filter_result(data, result)
        if shuffle:
            print("Shuffling the result")
            random.seed(42)
            random.shuffle(result)
        if idk:
            print("Overwriting the label with I don't know")
            result = [{"pred": "I don't know", "label": 0} for res in result]
        for i,frame in enumerate(data):
            gt_qa_pairs = parse_question_answers(frame['response_content'])
            pred_qa_pairs = [result.pop(0)["pred"] for qa_pair in gt_qa_pairs]
            data[i]["pred"] = pred_qa_pairs
    if limit_n is not None:
        data = data[:limit_n]

    if debug:
        data = data[:3]

    assert output_path is not None, "output_path must be specified when not in debug mode."

    if not click.confirm(f"About to grade {len(data)} examples. Continue?"):
        print("Aborting.")
        return

    if output_path.exists() and not click.confirm(f"{output_path} already exists. Overwrite?"):
        print("Aborting.")
        return

    with PoolWithTQDM(num_workers) as pool:
        results = []
        for graded_frame in tqdm(pool.imap(process_frame, data), total=len(data), ncols=120, desc="Processing frames"):
            results.append(graded_frame)
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

    save_list_of_dicts_to_csv(results, output_path.with_suffix(".csv"), ignored_columns=["observation", "pred"])
    get_avg_score(results)

    print(f"Grading complete. Results saved to {output_path} and {output_path.with_suffix('.csv')}")


if __name__ == '__main__':
    grade_vqa()  # pylint: disable=no-value-for-parameter
