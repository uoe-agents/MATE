import torch
import torch.nn as nn

from mate.algos.utils.models import build_sequential


class TaskEncoder(nn.Module):
    def __init__(self, obs_dim, act_dim, rews_dim, task_embedding_dim, hiddens, activation):
        super(TaskEncoder, self).__init__()
        self.act_dim = act_dim
        input_dim = obs_dim + act_dim + rews_dim
        self.rnn_hidden_dim = hiddens[0]

        if len(hiddens) > 1:
            # at least 2 hidden layers --> 1 hidden layer before RNN, rest after
            self.input_fc = build_sequential(input_dim, [hiddens[0]], activation, output_activation=True)
            self.rnn = nn.GRU(input_size=hiddens[0], hidden_size=hiddens[0], num_layers=1)
            self.output_fc = build_sequential(hiddens[0], hiddens[1:] + [task_embedding_dim], activation)
        else:
            # only 1 hidden layer --> first RNN before rest
            self.input_fc = None
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hiddens[0], num_layers=1)
            self.output_fc = build_sequential(hiddens[0], hiddens + [task_embedding_dim], activation)
        
    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.rnn_hidden_dim)

    def forward(self, obs, act, rew, hiddens=None):
    # def forward(self, obs, act, hiddens):
        x = torch.cat([obs, act, rew], dim=-1)

        if self.input_fc is not None:
            x = self.input_fc(x)

        # reshape to (seq_length, N, *)
        if len(x.shape) < 3:
            # sequence length missing
            seq_length = 1
            batch_size = x.shape[0]
            x_shape = x.shape[1:]
            x = x.unsqueeze(0)
            if hiddens is not None:
                hiddens_shape = hiddens.shape[1:]
                hiddens = hiddens.unsqueeze(0)
            else:
                hiddens_shape = None
        else:
            seq_length = None
            batch_size = None
            hiddens_shape = None
            x_shape = None

        output, final_hiddens = self.rnn(x, hiddens)

        # if reshaped before, unflatten again
        if seq_length is not None:
            if seq_length == 1:
                output = output.squeeze(0)
                final_hiddens = final_hiddens.squeeze(0)

        task_emb = self.output_fc(output)
        return task_emb, final_hiddens


class VAETaskEncoder(nn.Module):
    def __init__(self, obs_dim, act_dim, rew_dim, task_embedding_dim, hiddens, activation):
        super(VAETaskEncoder, self).__init__()
        input_dim = obs_dim + act_dim + rew_dim
        self.rnn_hidden_dim = hiddens[0]
        self.task_emb_dim = task_embedding_dim

        if len(hiddens) > 1:
            # at least 2 hidden layers --> 1 hidden layer before RNN, rest after
            self.input_fc = build_sequential(input_dim, [hiddens[0]], activation, output_activation=True)
            self.rnn = nn.GRU(input_size=hiddens[0], hidden_size=hiddens[0], num_layers=1)
            self.output_fc = build_sequential(hiddens[0], hiddens[1:] + [task_embedding_dim * 2], activation)
        else:
            # only 1 hidden layer --> first RNN before rest
            self.input_fc = None
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hiddens[0], num_layers=1)
            self.output_fc = build_sequential(hiddens[0], hiddens + [task_embedding_dim * 2], activation)
        
    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.rnn_hidden_dim)

    def reparameterise(self, mu, log_var):
        """
        Get VAE latent sample from distribution
        :param mu: mean for encoder's latent space
        :param log_var: log variance for encoder's latent space
        :return: sample of VAE distribution
        """
        # compute standard deviation from log variance
        std = torch.exp(0.5 * log_var)
        # get random sample with same dim as std
        eps = torch.randn_like(std)
        # sample from latent space
        sample = mu + (eps * std)
        return sample

    def forward(self, obs, act, rew, hiddens=None):
        x = torch.cat([obs, act, rew], dim=-1)

        if self.input_fc is not None:
            x = self.input_fc(x)

        # reshape to (seq_length, N, *)
        if len(x.shape) < 3:
            # sequence length missing
            seq_length = 1
            batch_size = x.shape[0]
            x_shape = x.shape[1:]
            x = x.unsqueeze(0)
            if hiddens is not None:
                hiddens_shape = hiddens.shape[1:]
                hiddens = hiddens.unsqueeze(0)
            else:
                hiddens_shape = None
        else:
            seq_length = None
            batch_size = None
            hiddens_shape = None
            x_shape = None

        output, final_hiddens = self.rnn(x, hiddens)

        # if reshaped before, unflatten again
        if seq_length is not None:
            if seq_length == 1:
                output = output.squeeze(0)
                final_hiddens = final_hiddens.squeeze(0)

        x = self.output_fc(output)

        # get mu and log_var from output
        if x.dim() > 2:
            mu = x[:, :, :self.task_emb_dim]
            log_var = x[:, :, self.task_emb_dim:]
        else:
            mu = x[:, :self.task_emb_dim]
            log_var = x[:, self.task_emb_dim:]

        # task embedding as concat of mu and log var
        task_emb = torch.cat([mu, log_var], dim=-1)

        # get sample from latent space
        z = self.reparameterise(mu, log_var)

        return task_emb, mu, log_var, z, final_hiddens


class TaskDecoder(nn.Module):
    def __init__(self, task_embedding_dim, obs_dim, act_dim, rew_dim, hiddens, activation):
        super(TaskDecoder, self).__init__()
        self.hiddens = hiddens
        self.obs_dim = obs_dim
        self.rew_dim = rew_dim
        self.decoder = build_sequential(task_embedding_dim + obs_dim + act_dim, hiddens + [obs_dim + rew_dim], activation)

    def forward(self, task_emb, obs, act):
        x = torch.cat([task_emb, obs, act], dim=-1)
        out = self.decoder(x)
        if out.dim() == 2:
            obs_pred = out[:, :self.obs_dim]
            rew_pred = out[:, self.obs_dim:]
        elif out.dim() == 3:
            obs_pred = out[:, :, :self.obs_dim]
            rew_pred = out[:, :, self.obs_dim:]

        return obs_pred, rew_pred


class ProbabilisticTaskDecoder(nn.Module):
    def __init__(self, task_embedding_dim, obs_dim, act_dim, rew_dim, hiddens, activation):
        super(ProbabilisticTaskDecoder, self).__init__()
        self.hiddens = hiddens
        self.obs_dim = obs_dim
        self.rew_dim = rew_dim
        self.decoder = build_sequential(task_embedding_dim + obs_dim + act_dim, hiddens + [obs_dim * 2 + rew_dim * 2], activation)

    def forward(self, task_emb, obs, act):
        x = torch.cat([task_emb, obs, act], dim=-1)
        out = self.decoder(x)
        if out.dim() == 2:
            obs_mu_pred = out[:, :self.obs_dim]
            obs_logstd_pred = out[:, self.obs_dim: self.obs_dim * 2]
            rew_mu_pred = out[:, -2 * self.rew_dim: -self.rew_dim]
            rew_logstd_pred = out[:, -self.rew_dim:]
        elif out.dim() == 3:
            obs_mu_pred = out[:, :, :self.obs_dim]
            obs_logstd_pred = out[:, :, self.obs_dim: self.obs_dim * 2]
            rew_mu_pred = out[:, :, -2 * self.rew_dim: -self.rew_dim]
            rew_logstd_pred = out[:, :, -self.rew_dim:]

        return obs_mu_pred, obs_logstd_pred, rew_mu_pred, rew_logstd_pred


class MixingNetwork(nn.Module):
    def __init__(self, obs_dim, num_distributions, hiddens, activation):
        super(MixingNetwork, self).__init__()
        self.network = build_sequential(obs_dim, hiddens + [num_distributions], activation)

    def forward(self, obs):
        return self.network(obs)
