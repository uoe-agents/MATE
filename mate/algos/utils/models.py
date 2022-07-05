import numpy as np
import torch
from torch._C import Value
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F


def _init_layer(m):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)
        return m

def build_sequential(num_inputs, hiddens, activation="relu", output_activation=False):
    modules = []

    if activation == "relu":
        nonlin = nn.ReLU
    elif activation == "tanh":
        nonlin = nn.Tanh
    else:
        raise ValueError(f"Unknown activation option {activation}!")
    
    assert len(hiddens) > 0
    modules.append(_init_layer(nn.Linear(num_inputs, hiddens[0])))
    for i in range(len(hiddens) - 1):
        modules.append(nonlin())
        modules.append(_init_layer(nn.Linear(hiddens[i], hiddens[i + 1])))
    if output_activation:
        modules.append(nonlin())
    return nn.Sequential(*modules)


class Actor(nn.Module):
    def __init__(self, actor_input_dim, task_emb_dim, actor_output_dim, actor_hiddens, activation):
        super(Actor, self).__init__()
        input_dim = actor_input_dim
        if task_emb_dim:
            input_dim += task_emb_dim
        self.num_outputs = actor_output_dim
        self.actor = build_sequential(input_dim, actor_hiddens + [actor_output_dim], activation)
        self.recurrent = False

    def forward(self, inputs):
        raise NotImplementedError
    
    def init_hidden(self, batch_size=1):
        return None
    
    def _get_dist(self, actor_features):
        return Categorical(logits=actor_features)

    def act(self, inputs, task_emb, hiddens, deterministic=False):
        if task_emb is not None:
            inputs = torch.cat([inputs, task_emb], dim=-1)
        actor_features = self.actor(inputs)
        dist = self._get_dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action.unsqueeze(-1), hiddens

    def evaluate_actions(self, inputs, task_emb, hiddens, action):
        if task_emb is not None:
            inputs = torch.cat([inputs, task_emb], dim=-1)
        actor_features = self.actor(inputs)
        dist = self._get_dist(actor_features)

        action_log_probs = dist.log_prob(action.squeeze()).unsqueeze(-1)
        dist_entropy = dist.entropy().mean()

        return action_log_probs, dist_entropy, hiddens
    
    def evaluate_policy_distribution(self, inputs, task_emb, hiddens):
        if task_emb is not None:
            inputs = torch.cat([inputs, task_emb], dim=-1)
        actor_features = self.actor(inputs)
        dist = self._get_dist(actor_features)

        policy_probs = []
        for a in range(self.num_outputs):
            actions = torch.ones(inputs.shape[0],).to(inputs.device) * a
            action_log_probs = dist.log_prob(actions)
            policy_probs.append(action_log_probs)
        policy_log_probs = torch.stack(policy_probs, dim=1).squeeze()

        return policy_log_probs, hiddens


class RecurrentActor(nn.Module):
    def __init__(self, actor_input_dim, task_emb_dim, actor_output_dim, actor_hiddens, activation):
        super(RecurrentActor, self).__init__()
        input_dim = actor_input_dim
        if task_emb_dim:
            input_dim += task_emb_dim
        self.num_outputs = actor_output_dim
        self.actor_hiddens = actor_hiddens
        self.rnn_hidden_dim = actor_hiddens[0]
        self.recurrent = True

        if len(actor_hiddens) > 1:
            # at least 2 hidden layers --> 1 hidden layer before RNN, rest after
            self.input_fc = build_sequential(input_dim, [actor_hiddens[0]], activation, output_activation=True)
            self.rnn = nn.GRUCell(actor_hiddens[0], actor_hiddens[0])
            self.output_fc = build_sequential(actor_hiddens[0], actor_hiddens[1:] + [actor_output_dim], activation)
        else:
            # only 1 hidden layer --> first RNN before rest
            self.input_fc = None
            self.rnn = nn.GRUCell(input_dim, actor_hiddens[0])
            self.output_fc = build_sequential(actor_hiddens[0], actor_hiddens + [actor_output_dim], activation)

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.rnn_hidden_dim)

    def _actor_base(self, inputs, task_emb, hiddens):
        if task_emb is not None:
            inputs = torch.cat([inputs, task_emb], dim=-1)
        if self.input_fc is not None:
            x = self.input_fc(inputs)
        else:
            x = inputs
        # flatten hiddens if needed
        if len(hiddens.shape) > 2:
            hiddens_shape = hiddens.shape
            hiddens = hiddens.reshape(-1, self.rnn_hidden_dim)
            x = x.reshape(-1, x.shape[-1])
        else:
            hiddens_shape = None
        hiddens = self.rnn(x, hiddens)
        # if flattened before, unflatten again
        if hiddens_shape is not None:
            hiddens = hiddens.view(hiddens_shape)
        x = self.output_fc(hiddens)
        return x, hiddens

    def forward(self, inputs):
        raise NotImplementedError
    
    def _get_dist(self, actor_features):
        return Categorical(logits=actor_features)

    def act(self, inputs, task_emb, hiddens, deterministic=False):
        actor_features, hiddens = self._actor_base(inputs, task_emb, hiddens)
        dist = self._get_dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action.unsqueeze(-1), hiddens

    def evaluate_actions(self, inputs, task_emb, hiddens, action):
        actor_features, hiddens = self._actor_base(inputs, task_emb, hiddens)
        dist = self._get_dist(actor_features)

        action_log_probs = dist.log_prob(action.squeeze()).unsqueeze(-1)
        dist_entropy = dist.entropy().mean()

        return action_log_probs, dist_entropy, hiddens
    
    def evaluate_policy_distribution(self, inputs, task_emb, hiddens):
        actor_features, hiddens = self._actor_base(inputs, task_emb, hiddens)
        dist = self._get_dist(actor_features)

        policy_probs = []
        for a in range(self.num_outputs):
            actions = torch.ones(inputs.shape[0],).to(inputs.device) * a
            action_log_probs = dist.log_prob(actions)
            policy_probs.append(action_log_probs)
        policy_log_probs = torch.stack(policy_probs, dim=1).squeeze()

        return policy_log_probs, hiddens
    

class Critic(nn.Module):
    def __init__(self, critic_input_dim, task_emb_dim, critic_hiddens, activation):
        super(Critic, self).__init__()
        input_dim = critic_input_dim
        if task_emb_dim:
            input_dim += task_emb_dim
        self.critic = build_sequential(input_dim, critic_hiddens + [1], activation)

    def forward(self, inputs, task_emb):
        if task_emb is not None:
            inputs = torch.cat([inputs, task_emb], dim=-1)
        return self.critic(inputs)
