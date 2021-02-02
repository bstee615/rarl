sbatch -t1-00:00:00 --cpus-per-task=8 --mem=32G --output=logs/job.%J.out --error=logs/job.%J.err --job-name="$1" --nodes=1 $@
