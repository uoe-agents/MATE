from abc import ABC, abstractmethod

from gym.spaces import flatdim

from mate.algos.utils.utils import flatten

class AutoEncoder(ABC):
    def __init__(
        self,
        observation_space,
        action_space,
        cfg,
        **kwargs,
    ):
        self.n_agents = len(observation_space)
        self.config = cfg
        self.observation_dims = [flatdim(obs_space) for obs_space in observation_space]
        self.action_dims = [flatdim(act_space) for act_space in action_space]

        self.hiddens = None

        # set all values from config as attributes
        for k, v in flatten(cfg).items():
            setattr(self, k, v)
            
    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def restore(self, path):
        pass
    
    @abstractmethod
    def hidden_dims(self):
        """
        Get hidden dimensions for all encoders (potentially single or multiple)
        :return List[int]: dimension of hidden states of all encoders
        """
        return

    @abstractmethod
    def encode(self, obss, acts, rews, hiddens, no_grads=False):
        """
        Encode task embedding

        :param obss: observation of each agent (num_agents, parallel_envs, obs_space)
        :param acts: action of each agent (num_agents, parallel_envs)
        :param rews: reward of each agent (num_agents, parallel_envs)
        :param hiddens: hiddens of all encoders (num_encoders, parallel_envs, hidden_dim)
        :param no_grads: boolean whether no gradients should be computed
        :return: task embedding for all agents (num_agents) x (parallel_envs, task_emb_dim),
            hiddens for all encoders (num_encoders, parallel_envs, hidden_dim)
        """
        return

    @abstractmethod
    def update(self, obss, acts, hiddens, rews, next_obss, done_mask):
        """
        Update encoder and decoder
        :param obss: observations for each agent (n_agents) x (n_step, parallel_envs, obs_space)
        :param acts: actions for each agent (n_agents) x (n_step, parallel_envs)
        :param hiddens: hidden states for each encoder (num_encoders) x (n_step, parallel_envs, hidden_dim)
        :param rews: rewards for each agent (n_agents) x (n_step, parallel_envs)
        :param next_obss: observations for each agent (n_agents) x (n_step, parallel_envs, obs_space)
        :param done_mask: batch of done masks (joint for all agents) (n_step, parallel_envs)
        :return: loss dictionary
        """
        return
