#!/bin/bash

module load miniconda3
source activate ~/benjis/weile-lab/envs/rarl
cd ~/benjis/weile-lab/rarl
python3 main.py --verbose --name='rarl-default'
