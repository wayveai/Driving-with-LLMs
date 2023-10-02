import datetime
import os
import pickle
import re
import tempfile
from typing import Any, Dict, List

import fire
import numpy as np
import transformers
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from models.vector_bc import VectorBC
from utils.training_utils import get_control_lateral, get_control_longitudinal


class TrainerWithEval(Trainer):
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        prediction_loss_only = False
        # call super class method to get the eval outputs
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )  # log the prediction distribution using `wandb.Histogram` method.
        mean_abs_err = np.mean(np.abs(eval_output.predictions - eval_output.label_ids), axis=0)
        # Get the predicted TL classes
        predicted_classes = np.argmax(eval_output.predictions[:, :5], axis=1)
        # Get the actual TL classes
        actual_classes = np.argmax(eval_output.label_ids[:, :5], axis=1)
        # Compute the accuracy
        tl_accuracy = np.mean(predicted_classes == actual_classes)
        print(f"tl_accuracy: {tl_accuracy}")
        self.log({"tl_accuracy": float(tl_accuracy)})
        self.log({"tl_distance": float(mean_abs_err[5])})
        self.log({"car_error": float(mean_abs_err[6])})
        self.log({"ped_error": float(mean_abs_err[7])})
        self.log({"lon_control_error": float(mean_abs_err[8])})
        self.log({"lat_control_error": float(mean_abs_err[9])})
        return eval_output


def load_bc_dataset(data_path: str, shuffle=False):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    data_dict: Dict[str, List[Any]] = {
        'frame_num': [],
        'action_label': [],
        'route_descriptors': [],
        'vehicle_descriptors': [],
        'pedestrian_descriptors': [],
        'ego_vehicle_descriptor': [],
    }
    for d in data:
        data_dict['frame_num'].append(d['frame_num'])
        action_prompt = '\n'.join(d['input_prompt'].split('\n')[-4:])
        lon_control = get_control_longitudinal(action_prompt)
        lat_control = get_control_lateral(action_prompt)
        perception_labels = extract_labels(d['input_prompt'])
        data_dict['action_label'].append(perception_labels + [lon_control, lat_control])
        obs_dict = d['observation']
        data_dict['route_descriptors'].append(obs_dict['route_descriptors'])
        data_dict['vehicle_descriptors'].append(obs_dict['vehicle_descriptors'])
        data_dict['pedestrian_descriptors'].append(obs_dict['pedestrian_descriptors'])
        data_dict['ego_vehicle_descriptor'].append(obs_dict['ego_vehicle_descriptor'])
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.shuffle(seed=42) if shuffle else dataset
    return dataset


def extract_labels(text):
    def extract_number(pattern):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else 0.0

    def get_tl_label(text):
        label_map = {'no traffic lights': 0, 'red+yellow': 1, ' red': 2, ' green': 3, ' yellow': 4}
        one_hot = [0, 0, 0, 0, 0]
        for phrase, value in label_map.items():
            if phrase in text:
                one_hot[value] = 1
                break
        return one_hot

    labels = get_tl_label(text)
    label_patterns = [
        r"It is (\d+(?:\.\d+)?)m ahead",  # tl distance
        r"observing (\d+(?:\.\d+)?) cars",  # car label
        r"and (\d+(?:\.\d+)?) pedestrians",  # ped label
    ]
    for pattern in label_patterns:
        labels.append(extract_number(pattern))
    return labels


def train(
    # model/data params
    data_path: str = "data/vqa_train_10k.pkl",
    val_data_path: str = "data/vqa_test_1k.pkl",
    # training hyperparams
    batch_size: int = 128,
    num_epochs: int = 25,
    learning_rate: float = 1e-4,
    eval_steps: int = 79,
    wandb_project: str = "sim_llm",
    wandb_run_name: str = "",
    wandb_watch: str = "false",  # options: false | gradients | all
    wandb_log_model: str = "false",  # options: false | true
    resume_from_checkpoint: str = None,  # always resume from pre-finetuned alpaca model
    output_dir: str = None,
    mode: str = "train",
):
    transformers.set_seed(42)
    if output_dir is None:
        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = tempfile.mkdtemp(prefix=f"vector_bc_{current_timestamp}_")
    print(
        f"Training Vector BC with params:\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # Initialize model
    model = VectorBC(num_action_queries=6)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}, trainable params: {trainable_params}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    train_data = load_bc_dataset(data_path, shuffle=True)
    val_data = load_bc_dataset(val_data_path)

    # Initialize trainer
    trainer = TrainerWithEval(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            warmup_ratio=0.04,
            lr_scheduler_type="cosine",
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=2,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
    )

    if mode == "train":
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print("🤗🤗🤗🤗🤗Model saved to:", output_dir, "🤗🤗🤗🤗🤗")
        trainer.save_model()
    elif mode == "eval":
        trainer.evaluate()


if __name__ == "__main__":
    fire.Fire(train)
