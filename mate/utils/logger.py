from abc import ABC, abstractmethod
from hashlib import sha256
import json
import logging
import time
from typing import Dict

from omegaconf import OmegaConf, DictConfig


logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)


class Logger(ABC):
    def __init__(self, cfg: DictConfig):
        self.config = OmegaConf.to_container(cfg)

        non_hash_keys = ["seed"]
        self.config_hash = sha256(
            json.dumps(
                {k: v for k, v in self.config.items() if k not in non_hash_keys},
                sort_keys=True,
            ).encode("utf8")
        ).hexdigest()[-10:]

        self.wandb = None
        self.database = None

        self.last_time = time.time()
        self.last_update = 0
        self.last_step = 0

    def info(self, *args, **kwargs):
        logging.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        logging.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        logging.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        logging.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        logging.critical(*args, **kwargs)
    
    def watch(self, model):
        self.debug(model)
    
    @abstractmethod
    def log_metrics(self, d: Dict, step_name: str, step: int):
        ...
    
    def completed_run(self):
        pass

    def failed_run(self):
        pass
    
    def log_progress(
        self, infos, update, step, total_steps,
    ):
        elapsed = time.time() - self.last_time
        self.last_time = time.time()
        ups = (update - self.last_update) / elapsed
        self.last_update = update
        steps_elapsed = step
        fps = (steps_elapsed - self.last_step) / elapsed
        self.last_step = step

        self.info(f"Updates {update}, Environment timesteps {steps_elapsed}")
        self.info(
            f"UPS: {ups:.1f}, FPS: {fps:.1f}, {steps_elapsed}/{total_steps} ({100 * steps_elapsed/total_steps:.2f}%) completed"
        )
        if infos:
            mean_return = sum([info["episode_reward"].sum() for info in infos]) / len(infos)
            self.info(f"Last {len(infos)} episodes with mean return: {mean_return:.3f}")
        self.info("-------------------------------------------")

    def log_episode(self, timestep, info, main_label="Train", print_train_log=False):
        info["episode_reward"] = sum(info["episode_reward"])
        if "terminal_observation" in info:
            del(info["terminal_observation"])
        log_dict = {}
        for k, v in info.items():
            log_dict[f"{main_label}/{k.replace('/','_')}"] = v
        self.log_metrics(log_dict, "timestep", timestep)
        if main_label == "Train":
            if print_train_log:
                self.info(
                    f"Completed episode {info['completed_episodes']}: Steps = {info['episode_length']} / Total Return = {info['episode_reward']:.3f} / Total duration = {info['episode_time']}s"
                )
        else:
            self.info(
                f"Completed evaluation: Steps = {info['episode_length']} / Total Return = {info['episode_reward']:.3f} / Total duration = {info['episode_time']}s"
            )
    
    
class PrintLogger(Logger):
    def __init__(self, cfg):
        super(PrintLogger, self).__init__(cfg)
    
    def log_metrics(self, d: Dict, step_name: str, step: int):
        self.info(f"---------- {step_name} = {step} -----------")
        for k, v in d.items():
            self.info(f"\t{k} = {v}")
        self.info("")


class TensorboardLogger(Logger):
    def __init__(self, cfg, tensorboard_dir):
        super(TensorboardLogger, self).__init__(cfg)
        from torch.utils.tensorboard import SummaryWriter
        self.tensorboard_logger = SummaryWriter(tensorboard_dir)

    def log_metrics(self, d: Dict, step_name: str, step: int):
        for key, v in d.items():
            self.tensorboard_logger.add_scalar(key, v, step)
