#!/bin/bash

module load miniconda3
source activate /work/LAS/weile-lab/benjis/weile-lab/envs/rarl
cd /work/LAS/weile-lab/benjis/weile-lab/rarl
python3 main.py --control --name=original-lerrel-big
