#!/usr/bin/env bash
#SBATCH --job-name=JOB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.gallo@phd.units.it
#SBATCH --account=mgallo02@login03
#SBATCH --partition=g100_all_serial
#SBATCH --qos=noQOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=./log_opt/%x.o%A-%a
#SBATCH --error=./log_opt/%x.o%A-%a



python -u try.py