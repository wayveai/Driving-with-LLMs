# pylint: skip-file
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import click
import openai
from retry import retry
from tqdm import tqdm


def make_context():
    prompt = f"""I am a certified professional driving instructor and I am currently demonstrating driving in a residential area of London to a student.
In London, we drive on the left side of road according to the UK driving rules.
A red+yellow signal on a traffic light is typically used as a warning signal to indicate that the light is about to turn green.
As we drive, I observe multiple objects such as vehicles, pedestrians, and traffic lights in front of us.
For each object, I am aware of their direction and distance from our car, which I measure in degrees, with 0 indicating it's right in front of us, 90 indicating it's to the right, and -90 indicating it's on the left.
This means that negative angles indicate that object is to the left of us and positive angles that its to the right of us.
If angle is larger than 90 degrees, its a sharp angle: e.g 134 degrees is a sharp right, -150 degrees is a sharp left.
If a car is driving in an opposite direction and is to the right of us, it is driving on an opposite lane.
I'm also driving in a defensive way and I'm paying varying levels of attention to each object (I measure it in percentage from 0% to 100%) depending on how they might be a hazard that may cause me to change speed, direction, stop, or even cause harm to myself.

Now design 16 random question and answer pairs that the student might ask about the current driving scenario. The answers should based on the current given input observations and your reasonings. Ask diverse questions, and give corresponding answers.

Format each QA pair in a single line as a JSON dictionary like {{"question": "xxx", "answer": "xxx"}}. Only output 16 lines of single-line JSON. Do not include any other explanation.
Must include these 6 questions, but please rephase the question to a more natural way:
- What are you seeing/observing
- What are you paying attention to and why
- Are there any traffic light / what's the color of the traffic light
- What's your current speed and steering angle / current state
- What is your action and why / how are you going to drive in this situation and why
- Summarize the current driving scenario in high level / describe the current situation

When asked about the action, always return the answer in this way:
```
Here are my actions:
- Accelerator pedal 0%
- Brake pedal 91%
- Steering 31% to the left.

Reason:
Because...
```
Also include one driving related question that cannot be observed in the observation input, and answer something like "I'm unable to answer this question based on the observation I have", and then describe your input observation

Also include one random question that is not related to driving, and answer something like "As an AI Driver, the question you asked is out of my scope, but I can try answer it" and then answer the question normally.
"""
    return prompt


@retry(tries=5, delay=2, backoff=2)
def make_description_from_prompt(input_prompt):
    context = make_context()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": input_prompt},
        ],
        temperature=1.0,
    )
    first_response = response["choices"][0]["message"]["content"]
    return first_response


def process_frame(frame_num, observation, input_prompt, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
        print("loading from cache: ", frame_num)
        return result

    print("making description for frame: ", frame_num)
    response_content = make_description_from_prompt(input_prompt)
    result = {
        "frame_num": frame_num,
        "observation": observation,
        "input_prompt": input_prompt,
        "response_content": response_content,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result


@click.command()
@click.option(
    "-i",
    "--input_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to pickle dataset file",
)
@click.option(
    "-o",
    "--output_folder",
    type=click.Path(dir_okay=True, path_type=Path),
    help="Path to json file",
)
@click.option("--max_steps", default=None, type=int)
@click.option("--stride", default=1, type=int)
@click.option("--num_process", default=1, type=int)
@click.option("--openai_api", required=True, type=str)
def main(
    input_path,
    output_folder,
    max_steps,
    stride,
    num_process,
    openai_api,
):
    # Init openai api key
    openai.api_key = openai_api

    cached_input_filename = os.path.expanduser(input_path)
    output_folder = os.path.expanduser(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    with open(cached_input_filename, "rb") as f:
        data = pickle.load(f)

    max_steps = len(data) if max_steps is None else max_steps
    frame_nums = range(0, max_steps, stride)

    args_list = [
        (
            frame_num,
            data[frame_num]["observation"],
            data[frame_num]["input_prompt"],
            os.path.join(output_folder, f"tmp_{frame_num}.pkl"),
        )
        for frame_num in frame_nums
    ]

    with Pool(num_process) as pool:
        results = list(
            tqdm(
                pool.starmap(process_frame, args_list),
                total=len(frame_nums),
                desc="Processing frames",
            )
        )

    cached_labels_filename = os.path.join(output_folder, "labeled_dataset.pkl")
    with open(cached_labels_filename, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Results saved to ", cached_labels_filename)


if __name__ == "__main__":
    main()
