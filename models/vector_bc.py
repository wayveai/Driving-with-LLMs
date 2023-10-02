import torch
import torch.nn as nn
from torch.nn import MSELoss

from utils.vector_utils import VectorObservation, VectorObservationConfig
from models.mlp import MLP
from models.transformer import Perceiver
from models.vector_encoder import VectorEncoder, VectorEncoderConfig


class VectorBC(nn.Module):
    def __init__(self, num_vector_tokens=64, num_action_queries=6, num_blocks=18):
        super().__init__()
        encoder_config = VectorEncoderConfig()
        self.num_vector_tokens = num_vector_tokens
        self.vector_encoder = VectorEncoder(
            encoder_config, VectorObservationConfig(), num_vector_tokens
        )

        self.policy = Perceiver(
            model_dim=encoder_config.model_dim,
            context_dim=encoder_config.model_dim,
            num_latents=num_vector_tokens,
            num_blocks=num_blocks,
            num_heads=encoder_config.num_heads,
            num_queries=num_action_queries,
        )
        self.tl_class_head = MLP(encoder_config.model_dim, (256, 256), 5)
        self.tl_d_head = MLP(encoder_config.model_dim, (256, 256), 1)
        self.car_head = MLP(encoder_config.model_dim, (256, 256), 1)
        self.ped_head = MLP(encoder_config.model_dim, (256, 256), 1)
        self.lon_act_head = MLP(encoder_config.model_dim, (256, 256), 1)
        self.lat_act_head = MLP(encoder_config.model_dim, (256, 256), 1)

    def forward(
        self,
        action_label,
        route_descriptors=None,
        vehicle_descriptors=None,
        pedestrian_descriptors=None,
        ego_vehicle_descriptor=None,
    ):
        # Create the vector observation
        vector_obs = VectorObservation(
            route_descriptors=route_descriptors,
            vehicle_descriptors=vehicle_descriptors,
            pedestrian_descriptors=pedestrian_descriptors,
            ego_vehicle_descriptor=ego_vehicle_descriptor,
        )
        encoder_output = self.vector_encoder(vector_obs)
        policy_output, _= self.policy(encoder_output, encoder_output)
        query_output = list(policy_output.unbind(dim=-2))
        # breakpoint()
        output_tl_class = torch.softmax(self.tl_class_head(query_output[0]), dim=-1)
        output_tl_d = self.tl_d_head(query_output[1])
        output_car = self.car_head(query_output[2])
        output_ped = self.ped_head(query_output[3])
        output_lon_act = self.lon_act_head(query_output[4])
        output_lat_act = self.lat_act_head(query_output[5])

        all_output = torch.cat(
            [output_tl_class, output_tl_d, output_car, output_ped, output_lon_act, output_lat_act],
            dim=-1,
        )
        loss_fct = MSELoss()
        loss = loss_fct(all_output, action_label)
        return {"loss": loss, "logits": all_output}
