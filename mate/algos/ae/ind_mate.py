import os

import numpy as np
import torch
from torch.distributions import Normal

from mate.algos.ae.autoencoder import AutoEncoder
from mate.algos.utils.ae_models import VAETaskEncoder, ProbabilisticTaskDecoder, TaskEncoder, TaskDecoder


class IndependentMATE(AutoEncoder):
    def __init__(
        self,
        observation_space,
        action_space,
        cfg,
        **kwargs,
    ):
        super(IndependentMATE, self).__init__(observation_space, action_space, cfg)
        if cfg.encoder.type == 'vae':
            self.encoders = [
                VAETaskEncoder(
                    obs_dim,
                    act_dim,
                    1,
                    cfg.task_emb_dim,
                    cfg.encoder.hiddens,
                    cfg.encoder.activation,
                ).to(cfg.device)
                for obs_dim, act_dim in zip(self.observation_dims, self.action_dims)
            ]
            # agents get mu + log_var as task embedding!
            self.task_emb_dim *= 2
        elif cfg.encoder.type == 'deterministic':
            self.encoders = [
                TaskEncoder(
                    obs_dim,
                    act_dim,
                    1,
                    cfg.task_emb_dim,
                    cfg.encoder.hiddens,
                    cfg.encoder.activation,
                ).to(cfg.device)
                for obs_dim, act_dim in zip(self.observation_dims, self.action_dims)
            ]
        else:
            raise ValueError(f"Unknown encoder type {cfg.encoder.type} not supported!")

        if cfg.decoder.type == 'probabilistic':
            self.decoder = ProbabilisticTaskDecoder(
                cfg.task_emb_dim,
                sum(self.observation_dims),
                sum(self.action_dims),
                self.n_agents,
                cfg.decoder.hiddens,
                cfg.decoder.activation,
            ).to(cfg.device)
        elif cfg.decoder.type == 'deterministic':
            self.decoder = TaskDecoder(
                cfg.task_emb_dim,
                sum(self.observation_dims),
                sum(self.action_dims),
                self.n_agents,
                cfg.decoder.hiddens,
                cfg.decoder.activation,
            ).to(cfg.device)
        else:
            raise ValueError(f"Unknown decoder type {cfg.decoder.type} not supported!")

        self.params = list(self.decoder.parameters())
        for encoder in self.encoders:
            self.params += list(encoder.parameters())
        self.optimiser = torch.optim.Adam(self.params, self.lr)

        self.saveables = {}
        for i, encoder in enumerate(self.encoders):
            self.saveables[f"encoder_{i+1}"] = encoder
        self.saveables[f"decoder"] = self.decoder
        self.saveables[f"optimiser"] = self.optimiser
    
    def save(self, path):
        torch.save(self.saveables, os.path.join(path, "ae_models.pt"))

    def restore(self, path):
        checkpoint = torch.load(os.path.join(path, "ae_models.pt"))
        for k, v in self.saveables.items():
            if k not in checkpoint:
                print(f"{k} not found in {path}")
                continue
            v.load_state_dict(checkpoint[k].state_dict())
    
    def hidden_dims(self):
        """
        Get hidden dimensions for all encoders (potentially single or multiple)
        :return List[int]: dimension of hidden states of all encoders
        """
        return [encoder.rnn_hidden_dim for encoder in self.encoders]
            
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
        task_embs = []
        hidden_list = []
        for obs, act, rew, hidden, encoder, act_dim in zip(obss, acts, rews, hiddens, self.encoders, self.action_dims):
            act_onehot = torch.nn.functional.one_hot(act.squeeze(-1).long(), act_dim).float()
            if no_grads:
                with torch.no_grad():
                    if self.encoder_type == 'vae':
                        task_emb, _, _, _, hiddens = encoder(obs, act_onehot, rew, hidden)
                    else:
                        task_emb, hiddens = encoder(obs, act_onehot, rew, hidden)
            else:
                if self.encoder_type == 'vae':
                    task_emb, _, _, _, hiddens = encoder(obs, act_onehot, rew, hidden)
                else:
                    task_emb, hidden = encoder(obs, act_onehot, rew, hidden)
            task_embs.append(task_emb)
            hidden_list.append(hidden)

        return task_embs, hidden_list

    def zero_grad(self):
        self.optimiser.zero_grad()
    
    def update(self, obss, acts, hiddens, rews, next_obss, done_mask):
        """
        Update encoder and decoder
        :param obss: observations for each agent (n_agents) x (n_step, parallel_envs, obs_space)
        :param acts: actions for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param hiddens: hidden states for each encoder (num_encoders) x (n_step, parallel_envs, hidden_dim)
        :param rews: rewards for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param next_obss: observations for each agent (n_agents) x (n_step, parallel_envs, obs_space)
        :param done_mask: batch of done masks (joint for all agents) (n_step, parallel_envs)
        :return: loss dictionary
        """
        act_onehots = [
            torch.nn.functional.one_hot(act.squeeze(-1).long(), act_dim).float()
            for act, act_dim in zip(acts, self.action_dims)
        ]

        # mask out entries where episode has terminates (do not make predictions across episodes)
        mask = done_mask == 1.0
        joint_obs = torch.concat(obss, dim=-1)[mask]
        joint_act_onehots = torch.concat(act_onehots, dim=-1)[mask]
        joint_rews = torch.concat(rews, dim=-1)[mask]
        joint_next_obss = torch.concat(next_obss, dim=-1)[mask]

        # multiply each vector in number of agents for total loss computation in single batch
        joint_obs = joint_obs.expand(self.n_agents, *joint_obs.shape)
        joint_act_onehots = joint_act_onehots.expand(self.n_agents, *joint_act_onehots.shape)
        joint_rews = joint_rews.expand(self.n_agents, *joint_rews.shape)
        joint_next_obss = joint_next_obss.expand(self.n_agents, *joint_next_obss.shape)

        task_embs = []
        kl_loss = 0
        for i, (obs, act_onehot, rew, hidden, encoder) in enumerate(zip(obss, act_onehots, rews, hiddens, self.encoders)):
            # mask out entries where episode has terminates (do not make predictions across episodes)
            obs = obs[mask]
            act_onehot = act_onehot[mask]
            rew = rew[mask]
            hidden = hidden[mask]

            # compute task embedding
            if self.encoder_type == 'vae':
                _, mu, log_var, z, _ = encoder(obs, act_onehot, rew, hidden)
                task_emb = z
                kl_loss += -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            else:
                task_emb, _ = encoder(obs, act_onehot, rew, hidden)
            task_embs.append(task_emb)
        task_embs = torch.stack(task_embs, dim=0)

        if self.decoder_type == 'probabilistic':
             # get prediction distributions
            joint_pred_obss_mu, joint_pred_obss_logstd, joint_pred_rews_mu, joint_pred_rews_logstd = self.decoder(task_embs, joint_obs, joint_act_onehots)
            obss_dist = Normal(joint_pred_obss_mu, joint_pred_obss_logstd.exp())
            rews_dist = Normal(joint_pred_rews_mu, joint_pred_rews_logstd.exp())

            # get log probabilities of actually encountered obs and reward
            # under predicted distributions
            obss_log_probs = obss_dist.log_prob(joint_next_obss)
            rews_log_probs = rews_dist.log_prob(joint_rews)

            # compute prediction losses as mean negative log probability
            obs_loss = -obss_log_probs.mean()
            rew_loss = -rews_log_probs.mean()
        else:
            # compute reconstruction loss as MSE of deterministic prediction
            joint_pred_obss, joint_pred_rews = self.decoder(task_embs, joint_obs, joint_act_onehots)
            obs_loss = (joint_next_obss - joint_pred_obss).pow(2).mean()
            rew_loss = (joint_rews - joint_pred_rews).pow(2).mean()

        # compute total loss
        loss = self.obs_loss_coef * obs_loss + self.rew_loss_coef * rew_loss
        if self.encoder_type == 'vae':
            loss += self.kl_loss_coef * kl_loss

        loss.backward()

        # gradient norm
        if self.max_grad_norm is not None and self.max_grad_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)

        self.optimiser.step()

        loss_dict = {
            "AE/obs_loss": obs_loss.item(),
            "AE/rew_loss": rew_loss.item(),
            "AE/loss": loss.item(),
        }

        if self.encoder_type == 'vae':
            loss_dict["AE/kl_loss"] = kl_loss.item()

        return loss_dict
