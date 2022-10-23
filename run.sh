#!/usr/bin/env bash
#
#SBATCH --job-name=try500
#SBATCH --output=er_%j.txt
#
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
##SBATCH --shared
#SBATCH -A ronnie-account
#SBATCH -p ronnieq
#SBATCH --mem=1000
##SBATCH --time=12:00:00
python newcheb.py --json_rep "input_files/input_report.json" --json_task "input_files/input_task_ut_ang_mom_H_2lvls_elect.json" > out.txt 2>err
