#!/bin/bash
for envname in Walker2D Hopper Ant HalfCheetah
do
env="Adversarial${envname}BulletEnv-v0"
bash batch.sh rarl.sh --name=original-big --env=$env --control
for p in 0.25 0.5 0.75 1.0
do
bash batch.sh rarl.sh --name=original-big_$p --env=$env --adv_percentage=$p
done
done
