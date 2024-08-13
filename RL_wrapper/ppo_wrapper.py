# a wrapper for PPO that manually close gripper when close to object
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np
import torch

import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor


class ActorCriticPolicyWrapperManualCloseGripper(ActorCriticPolicy):
    def __init__(self, 
                 
                 observation_space: spaces.Space,
                action_space: spaces.Space,
                lr_schedule: Schedule,
                net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                activation_fn: Type[nn.Module] = nn.Tanh,
                ortho_init: bool = True,
                use_sde: bool = False,
                log_std_init: float = 0.0,
                full_std: bool = True,
                use_expln: bool = False,
                squash_output: bool = False,
                features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                share_features_extractor: bool = True,
                normalize_images: bool = True,
                optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                optimizer_kwargs: Optional[Dict[str, Any]] = None,
                observation_to_track: tuple = (7,10),
                 close_threshold: float = 0.03,
                 manual_close_gripper_chance: float = 0.8):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.observation_to_track = observation_to_track
        self.close_threshold = close_threshold
        self.manual_close_gripper_chance = manual_close_gripper_chance

    def forward(self, obs, deterministic=False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)

        # if one of the obs is smaller than close_threshold, modify the action correspondingly
        # observations are in batch, so we need to iterate over each observation
        values_to_track = obs[:, self.observation_to_track[0]:self.observation_to_track[1]]
        values_norm = torch.linalg.norm(values_to_track, dim=1)
        assert values_norm.shape[0] == obs.shape[0]
        # values_norm = values_norm.reshape(-1, 1)
        # values_norm = values_norm.squeeze(0)
        # mask the values_norm with close_threshold
        close_to_object = values_norm < self.close_threshold

        # if close to object, set the last action to 1 with a certain probability
        random_close_gripper = torch.rand(close_to_object.shape).to(close_to_object.device)
        # print(random_close_gripper < self.manual_close_gripper_chance)

        gripper_height = obs[:, 16]
        gripper_height_threshold = 1.0

        mask = close_to_object & (random_close_gripper < self.manual_close_gripper_chance) & (gripper_height < gripper_height_threshold)
        # print(mask)

        actions[mask, -1] = 0.92
        # print(gripper_actions)
        
        
        # print(actions)
        

        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        
        return actions, values, log_prob


# class PPOManualCloseGripperWrapper(PPO):
#     def __init__(self, 
#                  observation_to_track: tuple = (7,10),
#                  close_threshold: float = 0.03,
#                  manual_close_gripper_chance: float = 0.8,
#                  *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.env = None

#     def set_env(self, env):
#         super().set_env(env)

#     def predict(self, obs, state=None, mask=None, deterministic=False):
#         action, state = super().predict(obs, state, mask, deterministic)
#         if self.env is not None:
#             if np.linalg.norm(obs[7:10]) < 0.02:
#                 action[-1] = 1
#         return action, state