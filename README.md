# Learning Task Embeddings for Teamwork Adaptation in Multi-Agent Reinforcement Learning

This repository is the official implementation of **multi-agent task embeddings (MATE)**

## Dependencies
Clone and install codebase with relevant dependencies using the provided `setup.py` with
```console
$ git clone <git-url>
$ cd <repo>
$ pip install -e .
```

We recommend to install dependencies in a virtual environment. Code was tested with Python 3.9.12.

To install the BoulderPush environment, run 
```console
$ git clone https://github.com/LukasSchaefer/boulderpush
$ cd boulderpush
$ pip install -e .
```

For our implementation, we defined new tasks within the multi-robot warehouse and multi-agent particle environments. To use these tasks, we provide modified code of the RWARE repository as well as modified code of PettingZoo which contains our tasks. To install, run
```console
$ https://github.com/LukasSchaefer/pettingzoo
$ cd PettingZoo
$ pip install -e .
```
and 
```console
$ git clone https://github.com/LukasSchaefer/robotic-warehouse
$ cd robotic-warehouse
$ pip install -e .
```


## Training
Within the `mate` directory execute

```console
$ python3 run.py +env=<ENV> +algorithm=maa2c seed=<SEED> +autoencoder=<MATE> logger=<LOGGER>
```
to train an algorithm in the specified environment. For logging, we provide a `tensorboard` and `print` logger which log into a tensorboard file or print out each logged metric in the console, respectively. For MATE, select
- cen_mate
- ind_mate
- mix_mate
for respective MATE paradigms used during training.

To adjust the training duration, model saving frequency, logging frequency and evaluation frequency, adjust the following configuration values (all in timesteps):
- algorithm.num_env_steps: total number of training timesteps
- algorithm.log_interval: interval at which logger reports progress in console
- algorithm.save_interval: interval at which MARL and MATE models are saved
- algorithm.eval_interval: interval at which evaluation episodes will be executed


## Fine-tuning
After training, model parameters are saved in the respective hydra output directory. Provide the additional `algorithm.load_dir=<PATH/TO/DIR/WITH/MODELS>` to load parameters for the agents and MATE to fine-tune in a new task. Note, the path needs to be provided as a **absolute** path.


## Codebase Structure

### Hydra Configurations
The interface of the main run script `run.py` is handled through [Hydra](https://hydra.cc/) with a hierarchy of configuration files under `configs/`.
These are structured in packages for

- MARL algorithms under `configs/algorithm/` (only MAA2C provided at the current time)
- environments under `configs/env/`
- MATE paradigms under `configs/autoencoder_algorithm/`
- hydra parameters under `configs/hydra/`
- logger parameters under `configs/logger/`
- default parameters in `configs/default.yaml`

## Citation
```
@inproceedings{schaefer2023mate,
  title={Learning Task Embeddings for Teamwork Adaptation in Multi-Agent Reinforcement Learning},
  author={Sch{\"a}fer, Lukas and Christianos, Filippos and Storkey, Amos and Albrecht, Stefano V.},
  booktitle={NeurIPS Workshop on Generalization in Planning},
  year={2023}
}
```
