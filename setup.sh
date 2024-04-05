#!/bin/bash
#conda create --name pytorch_env 
#eval "$(conda shell.bash hook)"
#conda activate pytorch_env
conda install --file requirements_pytorch_env.txt -c conda-forge -c pytorch -c nvidia

pip install 'asteroid==0.5.2' --no-dependencies
pip install 'comet-ml==3.32.8'  --no-deps

pip install 'tensorboardX==2.6' --no-dependencies
pip install soundfile --no-deps
pip install asteroid_filterbanks --no-deps
pip install huggingface-hub
pip install sortedcontainers
