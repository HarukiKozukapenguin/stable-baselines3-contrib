import gym
import torch as th
from torch import nn
import math

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)
        image_extract_dim = 8

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        # our case, we use 1d extractor for 1d depth image
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # Assume the image is single-channel (subspace.shape[0] == 0)
                n_input_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(
                    nn.Conv1d(n_input_channels, 8, 3, stride=1),
                    nn.Conv1d(8, 32, 3, stride=2),
                    nn.Conv1d(32, 8, 3, stride=1),
                    nn.Conv1d(8, 2, 3, stride=1),
                    nn.Flatten(),
                )
                flatten_dim = self.dim_calculation(subspace)
                extractors[key].add_module(
                    "linear", nn.Linear(flatten_dim, image_extract_dim)
                )
                total_concat_size += image_extract_dim
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], subspace.shape[0])
                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)

    def dim_calculation(self, subspace) -> int:
        return (
            math.floor((subspace.sphape[1] - (3 - 1) - 3) / 2) + 1 - (3 - 1) - (3 - 1)
        ) * 2
