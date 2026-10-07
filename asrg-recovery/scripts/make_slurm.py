"""Emit a SLURM array script that shards a config across N GPU jobs.

python scripts/make_slurm.py configs/blip2_full.yaml --shards 40 --hours 24 > job.sbatch && sbatch job.sbatch

Stage order matters for cost fidelity: run `--stage inject` everywhere first, then
`--stage recover`, so recovery timings never share a GPU with injection jobs.
Pin one GPU model (e.g. A100-80GB) for the whole study; GPU-hours are only comparable
on identical hardware (record it: nvidia-smi -L > runs/<name>/hardware.txt).
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--shards", type=int, default=20)
ap.add_argument("--hours", type=int, default=24)
ap.add_argument("--stage", default="all")
ap.add_argument("--partition", default="gpu")
ap.add_argument("--gres", default="gpu:a100:1")
a = ap.parse_args()
print(f"""#!/bin/bash
#SBATCH --job-name=asrg-{a.stage}
#SBATCH --array=0-{a.shards - 1}
#SBATCH --partition={a.partition}
#SBATCH --gres={a.gres}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={a.hours}:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
mkdir -p logs
nvidia-smi -L
python -m asrg_recovery.runner {a.config} --stage {a.stage} --shard ${{SLURM_ARRAY_TASK_ID}}/{a.shards}
""")
