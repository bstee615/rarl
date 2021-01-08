sbatch -t1-00:00:00 --gres=gpu:1 --cpus-per-task=24 --mem=64G --output=logs/job.%J.out --error=logs/job.%J.err --job-name="$1" --nodes=1 $1
