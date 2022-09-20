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
#SBATCH --mem=5000
##SBATCH --time=12:00:00
python newcheb.py --json_rep "input_report.json" --json_task "batch_jsons/input_task_ut_ang_mom_H_run3_var$1.json" > batch_jsons_out/out3_$1.txt 2>batch_jsons_out/err3_$1
