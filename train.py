# pylint: skip-file
import datetime
import os
import tempfile
from typing import List, Optional, Tuple

import fire
import numpy as np
import transformers
from peft import get_peft_model_state_dict  # noqa: E402
from transformers import logging  # noqa: F402

import wandb
from utils.model_utils import load_llama_tokenizer, load_model
from utils.training_utils import (
    DEFAULT_EVAL_ITEMS,
    decode_generation_seqeunces,
    eval_action,
    eval_tl,
    get_eval_distance_errors,
    get_train_val_data,
    log_txt_as_img,
)


class TrainerWithGeneration(transformers.Seq2SeqTrainer):
    """
    Custom Trainer class for sequence-to-sequence model with additional functionalities.
    Inherits from transformers.Seq2SeqTrainer.
    """

    def __init__(self, *args, **kwargs):
        self.vqa = kwargs.pop("vqa", False)
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs["data_collator"].tokenizer

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Overrided method to perform evaluation loop with custom eval and logging.
        """

        # ensure prediction loss is set to False
        prediction_loss_only = False

        # call parent class method to get the evaluation outputs
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # Perform additional operations based on evaluation output
        all_pred_tokens = (
            eval_output.predictions if self.vqa else eval_output.predictions[:, 77:]
        )  # remove the prompt for easier comparison
        all_pred = decode_generation_seqeunces(self.tokenizer, all_pred_tokens)
        all_label = decode_generation_seqeunces(self.tokenizer, eval_output.label_ids)
        print("all_pred", all_pred)
        print("all_label", all_label)

        if self.args.process_index != 0:
            return eval_output

        # Log the predictions
        if wandb.run is None:
            self.log({"i": None})  # dummy log to initialize wandb
        images = log_txt_as_img((512, 512), [all_pred[0], all_label[0]])
        wandb.log({"val_logits": wandb.Image(np.concatenate(images, axis=1))})
        wandb.log(
            {
                "val_results": wandb.Table(
                    columns=["pred", "label"],
                    data=[list(pair) for pair in zip(all_pred, all_label)],
                )
            }
        )

        # Evaluate traffic light
        tl_accuracy = eval_tl(all_pred, all_label)
        if tl_accuracy is not None:
            print(f"TL accuracy: {tl_accuracy}")
        else:
            print("No traffic light states found in predictions.")
        wandb.log({"tl_accuracy": tl_accuracy})
        eval_distance(
            all_pred, all_label, "tl_distance", r"It is (\d+(?:\.\d+)?)m ahead"
        )

        # Evaluate perceptions
        eval_distance(
            all_pred, all_label, "car_error", r"observing (\d+(?:\.\d+)?) cars"
        )
        eval_distance(
            all_pred, all_label, "ped_error", r"and (\d+(?:\.\d+)?) pedestrians"
        )

        # Evaluate actions
        average_error_lon, average_error_lat = eval_action(all_pred, all_label)
        if average_error_lon is not None and average_error_lat is not None:
            print(f"Average control error: {average_error_lon}, {average_error_lat}")
            wandb.log({"control_error_lon": average_error_lon})
            wandb.log({"control_error_lat": average_error_lat})
        return eval_output


def eval_distance(all_pred, all_label, label_name, pattern):
    distance_errors = get_eval_distance_errors(all_pred, all_label, pattern)
    if len(distance_errors) > 0:
        mean_error = np.mean(distance_errors)
        print(
            f"{label_name}: Mean Absolute Error (MAE): {mean_error}, Total num: {len(distance_errors)}"
        )
        wandb.log({label_name: mean_error})


def train(
    # model/data params
    base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
    data_path: str = "data/vqa_train_10k.pkl",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 32,
    num_epochs: int = 5,
    learning_rate: float = 3e-4,
    val_set_size: int = 1e6,
    eval_steps: int = 10,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
    group_by_length: bool = False,
    # wandb params
    wandb_project: str = "llm-driver",
    wandb_run_name: str = "",
    wandb_watch: str = "false",  # options: false | gradients | all
    wandb_log_model: str = "true",  # options: false | true
    resume_from_checkpoint: str = "models/weights/stage1_pretrained_model/",  # always resume from pre-finetuned model
    augment_times: int = 0,
    output_dir: Optional[str] = None,
    vqa: bool = False,
    eval_items: List[str] = DEFAULT_EVAL_ITEMS,
    mode: str = "train",
    load_pre_prompt_dataset: bool = False,
    val_data_path: str = "data/vqa_test_1k.pkl",
):
    if output_dir is None:
        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = tempfile.mkdtemp(prefix=f"lora-alpaca_{current_timestamp}_")

    if mode == "eval":
        transformers.set_seed(42)

    # set DDP flags
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print("Training Alpaca-LoRA model with params:")
        for k in [
            "base_model",
            "data_path",
            "output_dir",
            "batch_size",
            "micro_batch_size",
            "num_epochs",
            "learning_rate",
            "val_set_size",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "group_by_length",
            "wandb_project",
            "wandb_run_name",
            "wandb_watch",
            "wandb_log_model",
            "resume_from_checkpoint",
            "mode",
            "eval_items",
        ]:
            print(f"    {k}={eval(k)}")

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    model = load_model(
        base_model=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # Load tokenizer
    tokenizer = load_llama_tokenizer(base_model)

    train_data, val_data = get_train_val_data(
        data_path,
        tokenizer,
        val_data_path=val_data_path,
        val_set_size=val_set_size,
        augment_times=augment_times,
        load_pre_prompt_dataset=load_pre_prompt_dataset,
        vqa=vqa,
        eval_only=mode == "eval",
        eval_items=eval_items,
    )

    # Initialize trainer
    trainer = TrainerWithGeneration(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.04,
            lr_scheduler_type="cosine",
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=2,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            label_names=[
                "route_descriptors",
                "vehicle_descriptors",
                "pedestrian_descriptors",
                "ego_vehicle_descriptor",
                "user_input_ids",
                "user_attention_mask",
            ],
            prediction_loss_only=False,
            predict_with_generate=True,
            generation_max_length=384,
            generation_config=model.generation_config,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        vqa=vqa,
    )

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    logging.set_verbosity_info()
    if mode == "train":
        is_full_checkpoint = os.path.exists(
            os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        )
        trainer.train(resume_from_checkpoint=is_full_checkpoint)
        if local_rank == 0:
            print("🤗🤗🤗🤗🤗Model saved to:", output_dir, "🤗🤗🤗🤗🤗")
            model.save_pretrained(output_dir)
    elif mode == "eval":
        outputs = trainer.evaluate()
        print(outputs)


if __name__ == "__main__":
    import time

    st = time.time()
    fire.Fire(train)
    print("Total time:", time.time() - st)
