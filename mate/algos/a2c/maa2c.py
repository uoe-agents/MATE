import os

from gym.spaces import flatdim
import torch
import torch.nn as nn

from mate.algos.algorithm import Algorithm
from mate.algos.utils.models import Actor, RecurrentActor, Critic
from mate.algos.utils.utils import soft_update, concat_shapes


class MAA2C(Algorithm):
    def __init__(
        self,
        observation_space,
        action_space,
        cfg,
        task_emb_dim,
        **kwargs,
    ):
        super(MAA2C, self).__init__(observation_space, action_space, cfg, task_emb_dim)

        actor_fn = RecurrentActor if cfg.model.recurrent else Actor
        self.actors = [
            actor_fn(
                flatdim(obs_space),
                self.task_emb_dim,
                act_space.n,
                cfg.model.actor,
                cfg.model.activation,
            ).to(self.model_device)
            for obs_space, act_space in zip(observation_space, action_space)
        ]

        joint_obs_space = sum([flatdim(obs_space) for obs_space in observation_space])
        joint_obs_shape = concat_shapes([obs_space.shape for obs_space in observation_space])
        self.critics = [
            Critic(
                joint_obs_space,
                self.task_emb_dim,
                cfg.model.critic,
                cfg.model.activation,
            ).to(self.model_device)
            for _ in range(len(observation_space))
        ]
        self.target_critics = [
            Critic(
                joint_obs_space,
                self.task_emb_dim,
                cfg.model.critic,
                cfg.model.activation,
            ).to(self.model_device)
            for _ in range(len(observation_space))
        ]
        for target_critic, critic in zip(self.target_critics, self.critics):
            soft_update(target_critic, critic, 1.0)

        params = []
        for actor, critic in zip(self.actors, self.critics):
            params += list(actor.parameters())
            params += list(critic.parameters())
        self.optimiser = torch.optim.Adam(params, self.lr)

        self.saveables = {"optimiser": self.optimiser}
        for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            self.saveables[f"actor_{i+1}"] = actor
            self.saveables[f"critic_{i+1}"] = critic

    @property
    def parameters(self):
        params = []
        for actor, critic in zip(self.actors, self.critics):
            params += list(actor.parameters())
            params += list(critic.parameters())
        return params

    def save(self, path):
        torch.save(self.saveables, os.path.join(path, "models.pt"))

    def restore(self, path):
        checkpoint = torch.load(os.path.join(path, "models.pt"))
        for k, v in self.saveables.items():
            if k not in checkpoint:
                print(f"{k} not found in {path}")
                continue
            v.load_state_dict(checkpoint[k].state_dict())
    
    def hidden_dims(self):
        return [actor.rnn_hidden_dim if actor.recurrent else 1 for actor in self.actors]

    def act(self, obss, task_embs, hiddenss, evaluation=False):
        """
        Choose action for agent given observation (always uses stochastic policy greedy)

        :param obss: observation of each agent (num_agents, parallel_envs, obs_space)
        :param task_embs: task embeddings of each agent (num_agents, parallel_envs, task_emb_dim)
        :param hiddenss: hidden states of each agent (num_agents, parallel_envs, hidden_dim)
        :param evaluation: boolean whether action selection is for evaluation
        :return: actions (num_agents, parallel_envs, 1), hiddens (num_agents, parallel_envs, hidden_dim)
        """
        actions = []
        hiddens = []
        for obs, task_emb, hidden, actor in zip(obss, task_embs, hiddenss, self.actors):
            with torch.no_grad():
                action, hidden = actor.act(
                    obs,
                    task_emb,
                    hidden,
                    deterministic=evaluation if self.greedy_evaluation else False,
                )
                actions.append(action)
                hiddens.append(hidden)
        return actions, hiddens

    def _compute_returns(self, last_obs, last_task_emb, rew, done_mask):
        """
        Compute n-step returns for all agents
        :param last_obs: batch of observations at last step for each agent (n_agents) x (parallel_envs, obs_shape)
        :param last_task_emb: batch of task embeddings at last step for each agent (n_agents) x (parallel_envs, task_emb_dim)
        :param rew: batch of rewards for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param done_mask: batch of done masks (n_step + 1, parallel_envs, 1)
        """
        obs_shape = last_obs[0].shape[1:]
        joint_obs = torch.cat(last_obs, dim=-len(obs_shape))
        with torch.no_grad():
            next_value = [
                target_critic(joint_obs, last_z) for target_critic, last_z in zip(self.target_critics, last_task_emb)
            ]
        next_value = torch.stack(next_value)
        rew = torch.stack(rew)

        n_step = done_mask.shape[0] - 1
        returns = [next_value]
        for i in range(n_step - 1, -1, -1):
            ret = rew[:, i] + self.gamma * returns[0] * done_mask[i, :]
            returns.insert(0, ret)
        return torch.stack(returns[:-1], dim=1)

    def update(self, obs, act, rew, done_mask, task_embs, hiddens):
        """
        Compute and execute update
        :param obs: batch of observations for each agent (n_agents) x (n_step + 1, parallel_envs, obs_shape)
        :param act: batch of actions for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param rew: batch of rewards for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param done_mask: batch of done masks (joint for all agents) (n_step + 1, parallel_envs)
        :param task_embs: batch of task embeddings for each agent (n_agents) x (n_step, parallel_envs, task_emb_dim)
        :param hiddens: batch of hiddens for each agent (n_agents) x (n_step + 1, parallel_envs, hidden_dim)
        :return: dictionary of losses
        """
        done_mask = done_mask.unsqueeze(-1)

        # standardise rewards
        if self.standardise_rewards:
            rew = list(rew)
            for i in range(self.n_agents):
                rew[i] = (rew[i] - rew[i].mean()) / (rew[i].std() + 1e-5)

        last_task_embs = [z[-1] for z in task_embs] if task_embs is not None else [None for _ in range(self.n_agents)]
        returns = self._compute_returns([o[-1] for o in obs], last_task_embs, rew, done_mask)
        loss_dict = {}

        obs_shape = obs[0].shape[2:]
        joint_obs = torch.cat(obs, dim=-len(obs_shape))[:-1]

        self.optimiser.zero_grad()

        total_loss = 0

        for i in range(self.n_agents):
            actor = self.actors[i]
            critic = self.critics[i]

            agent_obs = obs[i][:-1]
            agent_act = act[i]

            agent_hidden = hiddens[i][:-1] if self.model_recurrent else None
            agent_task_emb = task_embs[i] if task_embs is not None else None

            agent_ret = returns[i]

            values = critic(joint_obs, agent_task_emb)
            action_log_probs, entropy, _ = actor.evaluate_actions(agent_obs, agent_task_emb, agent_hidden, agent_act)

            advantages = agent_ret - values

            value_loss = advantages.pow(2).mean()
            actor_loss = -(advantages.detach() * action_log_probs).mean()

            loss = (
                actor_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy
            )

            total_loss += loss

            loss_dict.update({
                f"agent_{i+1}/actor_loss": actor_loss.item(),
                f"agent_{i+1}/value_loss": value_loss.item(),
                f"agent_{i+1}/entropy": entropy.item(),
            })

        total_loss.backward()
        if self.max_grad_norm is not None and self.max_grad_norm != 0.0:
            nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.optimiser.step()

        # update target networks
        for critic, target_critic in zip(self.critics, self.target_critics):
            soft_update(target_critic, critic, self.tau)

        return loss_dict
