#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --array=1-2
#SBATCH --mem=60GB
#SBATCH --time=1-00:00:00
#SBATCH --job-name=cpu1
#SBATCH --qos=lhz
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc1,hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8,hpc9,vine3,vine4,vine6,vine11,vine12,rose7,rose8,rose9,lion17,lion3,rose1,rose[2-4]
#SBATCH --constraint="pascal|maxwell|turing|volta"
##SBATCH --constraint="gpu_24gb|gpu_32gb"

module load gcc-6.2.0
module load cuda-10.2

srun $(wandb agent lily/nurd-ood-jets/b9p9njvs)  # seed 0, RW, RW+JI