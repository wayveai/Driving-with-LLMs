# pylint: skip-file
from typing import List, Optional, Tuple, Union

import torch
from peft import PeftModelForCausalLM
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.vector_encoder import VectorEncoder, VectorEncoderConfig
from utils.vector_utils import VectorObservation, VectorObservationConfig


class LlamaForCausalLMVectorInput(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.weighted_loss_on_numbers = True
        if self.weighted_loss_on_numbers:
            number_tokens = [
                448,
                29900,
                29889,
                29896,
                29906,
                29941,
                29946,
                29945,
                29953,
                29955,
                29947,
                29929,
            ]  # -0.123456789
            weighted_mask = torch.ones(self.config.vocab_size)
            weighted_mask[number_tokens] = 3.0
            self.register_buffer("weighted_mask", weighted_mask)
        else:
            self.register_buffer("weighted_mask", None)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        route_descriptors=None,
        vehicle_descriptors=None,
        pedestrian_descriptors=None,
        ego_vehicle_descriptor=None,
        query_embeds=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        model_inputs.update(
            {
                "query_embeds": query_embeds,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        query_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Ingest vectors if in generation mode (query_embeds is not None)
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        elif inputs_embeds is None and input_ids is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if query_embeds is not None and past_key_values is None:
            inputs_embeds, attention_mask, _ = ingest_vectors(
                input_ids,
                inputs_embeds,
                query_embeds,
                attention_mask,
            )
            position_ids = None

        # from modeling_llama.py
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(weight=self.weighted_mask)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VectorLMWithLoRA(PeftModelForCausalLM):
    def __init__(self, model, peft_config, num_vector_tokens=64):
        super().__init__(model, peft_config)
        self.num_vector_tokens = num_vector_tokens
        self.vector_encoder = VectorEncoder(
            VectorEncoderConfig(), VectorObservationConfig(), num_vector_tokens
        )
        self.llm_proj = torch.nn.Linear(
            self.vector_encoder.out_features, self.config.hidden_size
        )
        self.to(model.device)
        self.modules_to_save = ["vector_encoder", "llm_proj"]
        self.generation_config = GenerationConfig(
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
            _from_model_config=False,
        )

    def embed_vector_and_prompt(
        self,
        input_ids,
        attention_mask,
        labels,
        route_descriptors,
        vehicle_descriptors,
        pedestrian_descriptors,
        ego_vehicle_descriptor,
    ):
        # Create the vector observation
        vector_obs = VectorObservation(
            route_descriptors=route_descriptors,
            vehicle_descriptors=vehicle_descriptors,
            pedestrian_descriptors=pedestrian_descriptors,
            ego_vehicle_descriptor=ego_vehicle_descriptor,
        )
        encoder_output = self.vector_encoder(vector_obs)
        inputs_vector = self.llm_proj(
            encoder_output
        )  # Adjust this line for multiple tokens

        # Generate token embeddings
        inputs_embeds = self.model.model.embed_tokens(input_ids)

        # Concatenate the vector embeddings with the token embeddings
        new_inputs_embeds, new_attention_mask, new_labels = ingest_vectors(
            input_ids,
            inputs_embeds,
            inputs_vector,
            attention_mask,
            labels,
        )

        return new_inputs_embeds, new_attention_mask, new_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        route_descriptors=None,
        vehicle_descriptors=None,
        pedestrian_descriptors=None,
        ego_vehicle_descriptor=None,
        **kwargs,  # those are 'user_input_ids', 'user_attention_mask'
    ):
        inputs_embeds, attention_mask, labels = self.embed_vector_and_prompt(
            input_ids,
            attention_mask,
            labels,
            route_descriptors,
            vehicle_descriptors,
            pedestrian_descriptors,
            ego_vehicle_descriptor,
        )

        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        loss = outputs.loss

        return {"loss": loss}

    def generate(self, **kwargs):
        route_descriptors = kwargs["route_descriptors"]
        vehicle_descriptors = kwargs["vehicle_descriptors"]
        pedestrian_descriptors = kwargs["pedestrian_descriptors"]
        ego_vehicle_descriptor = kwargs["ego_vehicle_descriptor"]

        vector_obs = VectorObservation(
            route_descriptors=route_descriptors,
            vehicle_descriptors=vehicle_descriptors,
            pedestrian_descriptors=pedestrian_descriptors,
            ego_vehicle_descriptor=ego_vehicle_descriptor,
        )
        encoder_output = self.vector_encoder(vector_obs)
        query_embeds = self.llm_proj(encoder_output)

        kwargs["query_embeds"] = query_embeds
        kwargs["input_ids"] = kwargs.pop("user_input_ids")
        kwargs["attention_mask"] = kwargs.pop("user_attention_mask")
        if "generation_config" not in kwargs:
            kwargs[
                "generation_config"
            ] = (
                self.generation_config
            )  # Override the generation config to make the padding tokens correct
        outputs = self.base_model.generate(**kwargs)
        return outputs


def ingest_vectors(
    input_ids, inputs_embeds, input_vectors, attention_mask, labels=None
):
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    vector_length = input_vectors.shape[1]
    # Find the position of the specific token sequence (10567 and 29901) for each instance in the batch
    token_sequence = torch.tensor([10567, 29901], device=input_ids.device)
    positions = (input_ids[:, :-1] == token_sequence[0]) & (
        input_ids[:, 1:] == token_sequence[1]
    )

    # Add 3 to get the vector insertion positions, and handle cases where the sequence is not found
    vector_input_positions = torch.argmax(positions.float(), dim=1) + 3
    vector_input_positions[vector_input_positions == 3] = 0
    vector_input_positions[vector_input_positions > seq_length] = seq_length
    # Create tensors to hold the updated inputs_embeds, attention_mask, and labels
    new_inputs_embeds = torch.zeros(
        batch_size,
        seq_length + vector_length,
        inputs_embeds.shape[2],
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    new_attention_mask = torch.zeros(
        batch_size,
        seq_length + vector_length,
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    new_labels = (
        torch.zeros(
            batch_size,
            seq_length + vector_length,
            device=labels.device,
            dtype=labels.dtype,
        )
        if labels is not None
        else None
    )
    for b in range(batch_size):
        vector_input_position = vector_input_positions[b]
        if vector_input_position == 0:
            vector_input_position = 1  # Insert the vector embeddings at position 1 if the token_sequence is not found (avoid the bos_token)
        new_inputs_embeds[b, :vector_input_position] = inputs_embeds[
            b, :vector_input_position
        ]
        new_inputs_embeds[
            b, vector_input_position : vector_input_position + vector_length
        ] = input_vectors[b]
        new_inputs_embeds[b, vector_input_position + vector_length :] = inputs_embeds[
            b, vector_input_position:
        ]

        new_attention_mask[b, :vector_input_position] = attention_mask[
            b, :vector_input_position
        ]
        new_attention_mask[
            b, vector_input_position : vector_input_position + vector_length
        ] = 1
        new_attention_mask[b, vector_input_position + vector_length :] = attention_mask[
            b, vector_input_position:
        ]

        if labels is not None:
            new_labels[b, :vector_input_position] = labels[b, :vector_input_position]
            new_labels[
                b, vector_input_position : vector_input_position + vector_length
            ] = -100
            new_labels[b, vector_input_position + vector_length :] = labels[
                b, vector_input_position:
            ]

    return new_inputs_embeds, new_attention_mask, new_labels
