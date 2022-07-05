import random

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    logger = hydra.utils.instantiate(cfg.logger, cfg=cfg)

    torch.set_num_threads(1)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
    else:
        logger.warning("No seed has been set.")

    if cfg.env.parallel_envs is not None:
        num_updates = cfg.algorithm.num_env_steps // (cfg.algorithm.n_steps * cfg.env.parallel_envs)
    else:
        num_updates = cfg.algorithm.num_env_steps // cfg.algorithm.n_steps

    env = hydra.utils.call(cfg.env, seed=cfg.seed)

    if cfg.autoencoder is not None:
        ae = hydra.utils.instantiate(
            cfg.autoencoder,
            observation_space=env.observation_space,
            action_space=env.action_space,
            cfg=cfg.autoencoder,
        )
        task_emb_dim = ae.task_emb_dim
    else:
        ae = None
        task_emb_dim = None

    alg = hydra.utils.instantiate(
        cfg.algorithm,
        observation_space=env.observation_space,
        action_space=env.action_space,
        cfg=cfg.algorithm,
        num_updates=num_updates,
        task_emb_dim=task_emb_dim,
    )

    try:
        hydra.utils.call(cfg, env_instance=env, alg_instance=alg, ae_instance=ae, logger_instance=logger, cfg=cfg)
    except (KeyboardInterrupt, AssertionError):
        logger.failed_run()
    else:
        logger.completed_run()

if __name__ == "__main__":
    main()
