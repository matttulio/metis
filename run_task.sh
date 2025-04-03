#!/bin/bash
#SBATCH -A=PRACE_IT
#SBATCH --job-name=JOB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.gallo@phd.units.it
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=./log_opt/%x.o%A-%a
#SBATCH --error=./log_opt/%x.o%A-%a



python -u try.py