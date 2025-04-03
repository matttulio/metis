#!/bin/bash
#SBATCH -A OGS23_PRACE_IT_0
#SBATCH --job-name=JOB
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=./log_opt/%x.o%A-%a
#SBATCH --error=./log_opt/%x.o%A-%a



python -u try.py