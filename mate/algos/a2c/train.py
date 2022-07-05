import os
from collections import deque
from pathlib import Path
import shutil

from gym.spaces import Box
import numpy as np
import torch

from mate.algos.utils.utils import split_batch, concat_shapes, split_dims


def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.remove("terminal_observation")

    for key in keys:
        values = [d[key] for d in info if key in d]
        mean = np.mean(values, 0)
        new_info[key] = mean

    return new_info


def save_models(alg, ae, logger, t):
    if logger.wandb:
        wandb_save_dir = os.path.join(logger.wandb.dir, "models")
        save_at = os.path.join(wandb_save_dir, f"t_{t}")
    else:
        save_at = os.path.join("models", f"t_{t}")

    os.makedirs(save_at, exist_ok=True)               
    alg.save(save_at)
    if ae:
        ae.save(save_at)

    if logger.database:
        logger.add_artifact(f"Models_{t}", save_at)


def evaluate(
    parallel_envs,
    envs,
    agent,
    ae,
    device,
    episodes_per_eval,
    split_obs,
):
    obs = envs.reset()
    n_agents = len(obs)
    hiddens = [
        torch.zeros(parallel_envs, hidden_dim).to(device) for hidden_dim in agent.hidden_dims()
    ]
    if ae:
        task_embs = [torch.zeros(parallel_envs, agent.task_emb_dim) for _ in range(n_agents)]
        ae_hiddens = [
            torch.zeros(parallel_envs, hidden_dim).to(device) for hidden_dim in ae.hidden_dims()
        ]
    else:
        task_embs = [None for _ in range(n_agents)]

    all_infos = []
    while len(all_infos) < episodes_per_eval:
        obs = split_obs(torch.cat([torch.from_numpy(o) for o in obs], dim=1).float())
        with torch.no_grad():
            actions, hiddens = agent.act(obs, task_embs, hiddens, evaluation=True)
        env_actions = torch.cat(actions, dim=1)
        n_obs, rew, done, infos = envs.step(env_actions.tolist())
        rew = list(torch.stack([torch.from_numpy(r).float() for r in rew], dim=-1).unsqueeze(-1))
        if ae:
            task_embs, ae_hiddens = ae.encode(obs, actions, rew, ae_hiddens, no_grads=True)

        obs = n_obs

        for i, (info, d) in enumerate(zip(infos, done)):
            if d:
                all_infos.append(info)
                for hidden in hiddens:
                    hidden[i, :].zero_()
                if ae:
                    for ae_hidden in ae_hiddens:
                        ae_hidden[i, :].zero_()

    return all_infos


def main(
    env_instance,
    alg_instance,
    ae_instance,
    logger_instance,
    cfg,
    **kwargs,
):
    envs = env_instance
    alg = alg_instance
    ae = ae_instance
    logger = logger_instance

    if cfg.algorithm.load_dir:
        # load params from given directory
        load_dir = cfg.algorithm.load_dir
        logger.info(f"Loading algorithm parameters from {load_dir}...")
        alg.restore(load_dir)
        if ae:
            logger.info(f"Loading MATE parameters from {load_dir}...")
            ae.restore(load_dir)

    obs = envs.reset()

    recurrent = cfg.algorithm.model.recurrent
    n_steps = cfg.algorithm.n_steps
    parallel_envs = cfg.env.parallel_envs
    n_agents = alg.n_agents
    num_updates = (
        int(cfg.algorithm.num_env_steps) // n_steps // parallel_envs
    )

    if cfg.autoencoder:
        mate_frozen = "frozen" in cfg.autoencoder and cfg.autoencoder.frozen
        if mate_frozen:
            logger.info("MATE parameters frozen throughout training!")
    else:
        mate_frozen = False

    # batches for n-step data
    # here we assume Box observation spaces with "shape" property
    observation_shapes = [obs_space.shape for obs_space in envs.observation_space]
    batch_obs = torch.zeros(n_steps + 1, parallel_envs, *concat_shapes(observation_shapes)).to(cfg.algorithm.model.device)
    batch_obs[0, :] = torch.cat([torch.from_numpy(o) for o in obs], dim=1)
    batch_done = torch.zeros(n_steps + 1, parallel_envs).to(cfg.algorithm.model.device)
    batch_act = torch.zeros(n_steps, parallel_envs, n_agents).to(cfg.algorithm.model.device)
    batch_rew = torch.zeros(n_steps, parallel_envs, n_agents).to(cfg.algorithm.model.device)
    if recurrent:
        batch_hiddens = torch.zeros(n_steps + 1, parallel_envs, sum(alg.hidden_dims())).to(cfg.algorithm.model.device)
    if ae:
        batch_ae_hiddens = torch.zeros(n_steps, parallel_envs, sum(ae.hidden_dims())).to(cfg.algorithm.model.device)
        batch_task_embs = torch.zeros(n_steps, parallel_envs, n_agents * ae.task_emb_dim).to(cfg.algorithm.model.device)
        split_task_emb = split_batch(n_agents * [ae.task_emb_dim], cfg.algorithm.model.device)
        split_ae_hiddens = split_batch([hidden_dim for hidden_dim in ae.hidden_dims()], cfg.algorithm.model.device)

    # define split functions from joint obs/ acts to individual ones
    # dimension of first dim of observation from back
    all_box = all([isinstance(space, Box) for space in envs.observation_space])
    if all_box:
        obs_split_dim = -len(envs.observation_space[0].shape)
    else:
        obs_split_dim = -1
    split_obs = split_batch(split_dims(envs.observation_space), cfg.algorithm.model.device, obs_split_dim)
    split_act = split_batch(n_agents * [1], cfg.algorithm.model.device)
    split_rew = split_batch(n_agents * [1], cfg.algorithm.model.device)
    if recurrent:
        split_hiddens = split_batch([hidden_dim for hidden_dim in alg.hidden_dims()], cfg.algorithm.model.device)

    all_infos = deque(maxlen=10)
    
    total_steps = 0
    completed_episodes = 0

    last_log_t = 0
    last_save_t = 0
    last_eval_t = 0

    for n_updates in range(1, num_updates + 1):
        # get n steps of data for each update
        for n in range(cfg.algorithm.n_steps):
            # Sample actions
            obs = split_obs(batch_obs[n, :])
            hiddens = split_hiddens(batch_hiddens[n, :, :]) if recurrent else [None for _ in range(n_agents)]
            if ae:
                task_embs, ae_hiddens = ae.encode(
                    split_obs(batch_obs[n-1, :]),
                    split_act(batch_act[n-1, :]),
                    split_rew(batch_rew[n-1, :]),
                    split_ae_hiddens(batch_ae_hiddens[n]),
                    # ae_hiddens,
                    no_grads=cfg.autoencoder.detach,
                )
            else:
                task_embs = [None for _ in range(n_agents)]

            actions, hiddens = alg.act(obs, task_embs, hiddens, evaluation=False)
            actions = torch.cat(actions, dim=1)

            next_obs, rewards, dones, infos = envs.step(actions.tolist())
            rewards = torch.from_numpy(rewards).float().to(cfg.algorithm.model.device)
            next_obs = [torch.from_numpy(o).float() for o in next_obs]
            next_obs_cat = torch.cat(next_obs, dim=1)
            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones]).squeeze().to(cfg.algorithm.model.device)

            # store step data
            batch_obs[n + 1, :] = next_obs_cat
            batch_act[n, :] = actions
            batch_done[n + 1, :] = masks
            batch_rew[n, :] = rewards
            if recurrent:
                batch_hiddens[n + 1, :] = torch.cat(hiddens, dim=-1)
            if ae:
                batch_task_embs[n, :] = torch.cat(task_embs, dim=-1)
            
            # log episode data for completed episodes
            for i, info in enumerate(infos):
                if info:
                    completed_episodes += 1
                    info["completed_episodes"] = completed_episodes
                    all_infos.append(info)
                    logger.log_episode(total_steps, info)
                    if recurrent:
                        batch_hiddens[n + 1, i, :].zero_()
                    if ae:
                        for ae_hidden in ae_hiddens:
                            ae_hidden[i, :].zero_()

            total_steps += cfg.env.parallel_envs

        if ae:
            # zero grad here to allow gradients from RL loss to backprop into encoder
            ae.zero_grad()

        loss_dict = alg.update(
            split_obs(batch_obs),
            split_act(batch_act),
            split_rew(batch_rew),
            batch_done,
            split_task_emb(batch_task_embs) if ae else None,
            split_hiddens(batch_hiddens) if recurrent else None,
        )
        loss_dict["updates"] = n_updates

        if ae and not mate_frozen:
            # train MATE if present unless specifically stated as frozen
            ae_loss_dict = ae.update(
                split_obs(batch_obs[:-1]),
                split_act(batch_act),
                split_ae_hiddens(batch_ae_hiddens),
                split_rew(batch_rew),
                split_obs(batch_obs[1:]),
                batch_done[1:],
            )
            loss_dict.update(ae_loss_dict)

        logger.log_metrics(loss_dict, "timestep", total_steps)

        batch_obs[0, :, :] = batch_obs[-1, :, :]
        batch_done[0, :] = batch_done[-1, :]
        if recurrent:
            batch_hiddens[0, :, :] = batch_hiddens[-1, :, :]
        if ae:
            batch_ae_hiddens[0, :, :] = torch.cat(ae_hiddens, dim=-1)
            batch_task_embs = torch.zeros(n_steps, parallel_envs, n_agents * ae.task_emb_dim).to(cfg.algorithm.model.device)

        if total_steps >= last_log_t + cfg.algorithm.log_interval and len(all_infos) > 1:
            logger.log_progress(all_infos, n_updates, total_steps, cfg.algorithm.num_env_steps)
            all_infos.clear()
            last_log_t = total_steps

        if cfg.algorithm.save_interval and (
            total_steps >= last_save_t + cfg.algorithm.save_interval
        ):
            save_models(alg, ae, logger, total_steps)
            last_save_t = total_steps

        if cfg.algorithm.eval_interval and (
            total_steps >= last_eval_t + cfg.algorithm.eval_interval or n_updates == num_updates
        ):
            all_infos = evaluate(
                parallel_envs,
                envs,
                alg,
                ae,
                cfg.algorithm.model.device,
                cfg.algorithm.episodes_per_eval,
                split_obs,
            )
            eval_info = _squash_info(all_infos)
            logger.log_episode(total_steps, eval_info, main_label="Eval")
            last_eval_t = total_steps

    # save models at very end
    save_models(alg, ae, logger, total_steps)

    envs.close()
