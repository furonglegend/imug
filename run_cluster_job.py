# scripts/run_cluster_job.py
"""
Small helper to generate a SLURM script for running the pipeline on an HPC cluster.
This file does not submit the job; it writes a script you can submit with `sbatch`.
"""
import textwrap
import os

SLURM_TEMPLATE = textwrap.dedent("""\
#!/bin/bash
#SBATCH --job-name=immuno_pipeline
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

module load cuda/11.8
source ~/venv/bin/activate

cd $SLURM_SUBMIT_DIR
python run_pipeline.py --config {config_path}
""")

def write_slurm_script(out_path='run_immuno_pipeline.sbatch', config_path='configs.py'):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(SLURM_TEMPLATE.format(config_path=config_path))
    print(f"Wrote SLURM script to {out_path}")
