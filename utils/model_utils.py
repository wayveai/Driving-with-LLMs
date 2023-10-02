import os
from typing import Tuple

import torch
from peft import LoraConfig, prepare_model_for_int8_training, set_peft_model_state_dict
from transformers import GenerationConfig, LlamaTokenizer

from models.vector_lm import LlamaForCausalLMVectorInput, VectorLMWithLoRA


def load_llama_tokenizer(base_model):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # Fix the decapoda-research tokenizer bug
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"  # Allow batched inference
    return tokenizer


def default_generation_config(**kwargs):
    return GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        use_cache=False,
        do_sample=True,
        max_length=384,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    )


def load_model(
    base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
    resume_from_checkpoint: str = "pretrained_model/",
    load_in_8bit: bool = True,
):
    # set DDP flags
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if ddp:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    # Initialize model
    llama_model = LlamaForCausalLMVectorInput.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="Vector_LM",
    )
    if load_in_8bit:
        llama_model = prepare_model_for_int8_training(llama_model)

        model = VectorLMWithLoRA(llama_model, lora_config)
        # We can load trainer state only from a full checkpoint
        is_full_checkpoint = os.path.exists(
            os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        )
        if is_full_checkpoint:
            # Check the available weights and load them
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location="cpu")
            model = set_peft_model_state_dict(model, adapters_weights)
        elif os.path.exists(resume_from_checkpoint):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            lora_checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model_lora.bin"
            )
            assert os.path.exists(checkpoint_name), "Checkpoint not found"
            if os.path.exists(lora_checkpoint_name):
                print(f"Loading LoRA model from {lora_checkpoint_name}")
                lora_weights = torch.load(lora_checkpoint_name, map_location="cpu")
                model = set_peft_model_state_dict(model, lora_weights)
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location="cpu")
            model = set_peft_model_state_dict(model, adapters_weights)
    else:
        model = VectorLMWithLoRA.from_pretrained(
            llama_model,
            resume_from_checkpoint,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

    model.config.use_cache = False  # insert vector not support cache now

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # Fix the generation of llama model
    model.generation_config = default_generation_config()
    model.model.model.generation_config = model.generation_config
    model.model.model.config.pad_token_id = 0
    model.model.model.config.bos_token_id = 1
    model.model.model.config.eos_token_id = 2

    return model
