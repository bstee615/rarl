#!/bin/bash
source /work/LAS/weile-lab/benjis/weile-lab/rarl/activate.sh
set -x
python3 main.py --log --verbose $@
