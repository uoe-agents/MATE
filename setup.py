from setuptools import find_packages, setup

setup(
    name='mate',
    packages=find_packages(),
    version='0.0.1',
    install_requires=[
        'click',
        'gym==0.21.0',
        'hydra-core==1.0.5',
        'lbforaging',
        'matplotlib',
        'omegaconf',
        'pyyaml==5.4.1',
        'sacred',
        'stable-baselines3',
        'tensorboard',
        'torch',
        'tqdm',
    ]
)
