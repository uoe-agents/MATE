from abc import ABC, abstractmethod

from gym.spaces.utils import flatdim

from mate.algos.utils.utils import flatten


class Algorithm(ABC):
    def __init__(
        self,
        observation_spaces,
        action_spaces,
        algorithm_config,
        task_emb_dim,
    ):
        self.n_agents = len(observation_spaces)
        self.obs_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.task_emb_dim = task_emb_dim
        self.config = algorithm_config

        self.obs_sizes = [flatdim(obs_space) for obs_space in observation_spaces]
        self.action_sizes = [flatdim(act_space) for act_space in action_spaces]

        # set all values from config as attributes
        for k, v in flatten(algorithm_config).items():
            setattr(self, k, v)
