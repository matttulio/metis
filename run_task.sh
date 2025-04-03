#!/bin/bash
#SBATCH -A=PRACE_IT
#SBATCH --job-name=JOB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.gallo@phd.units.it
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00



python -u try.py