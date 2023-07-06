from math import sqrt

import torch
import torch.nn as nn

from models.mlp import MLP
from models.transformer import Perceiver
from utils.vector_utils import VectorObservation, VectorObservationConfig


class VectorEncoderConfig:
    model_dim: int = 256
    num_latents: int = 32
    num_blocks: int = 7
    num_heads: int = 8


class VectorEncoder(nn.Module):
    def __init__(
        self,
        encoder_config: VectorEncoderConfig,
        observation_config: VectorObservationConfig,
        num_queries: int,
    ):
        super().__init__()

        model_dim = encoder_config.model_dim
        self.ego_vehicle_encoder = MLP(
            VectorObservation.EGO_DIM, [model_dim], model_dim
        )
        self.vehicle_encoder = MLP(
            VectorObservation.VEHICLE_DIM, [model_dim], model_dim
        )
        self.pedestrian_encoder = MLP(
            VectorObservation.PEDESTRIAN_DIM, [model_dim], model_dim
        )
        self.route_encoder = MLP(VectorObservation.ROUTE_DIM, [model_dim], model_dim)
        self.route_embedding = nn.Parameter(
            torch.randn((observation_config.num_route_points, model_dim))
            / sqrt(model_dim)
        )

        self.perceiver = Perceiver(
            model_dim=model_dim,
            context_dim=model_dim,
            num_latents=encoder_config.num_latents,
            num_blocks=encoder_config.num_blocks,
            num_heads=encoder_config.num_heads,
            num_queries=num_queries,
        )

        self.out_features = model_dim

    def forward(self, obs: VectorObservation):
        batch = obs.route_descriptors.shape[0]
        device = obs.route_descriptors.device

        route_token = self.route_embedding[None] + self.route_encoder(
            obs.route_descriptors
        )
        vehicle_token = self.vehicle_encoder(obs.vehicle_descriptors)
        pedestrian_token = self.pedestrian_encoder(obs.pedestrian_descriptors)
        context = torch.cat((route_token, pedestrian_token, vehicle_token), -2)
        context_mask = torch.cat(
            (
                torch.ones(
                    (batch, route_token.shape[1]), device=device, dtype=bool
                ),  # route
                obs.pedestrian_descriptors[:, :, 0] != 0,  # pedestrians
                obs.vehicle_descriptors[:, :, 0] != 0,  # vehicles
            ),
            dim=1,
        )

        ego_vehicle_state = obs.ego_vehicle_descriptor
        ego_vehicle_feat = self.ego_vehicle_encoder(ego_vehicle_state)

        feat, _ = self.perceiver(ego_vehicle_feat, context, context_mask=context_mask)
        feat = feat.view(
            batch,
            self.perceiver.num_queries,
            feat.shape[-1],
        )

        return feat
