#!/usr/bin/env bash
#
#SBATCH --job-name=try500
#SBATCH --output=../batch_jsons_out/er_%j.txt
#
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
##SBATCH --shared
#SBATCH -A ronnie-account
#SBATCH -p ronnieq
#SBATCH --mem=1000
##SBATCH --time=12:00:00
python ../newcheb.py --json_rep "input_report.json" --json_task "../batch_jsons/input_task_ut_ang_mom_H_run$2_var$1.json" > ../batch_jsons_out/out$2_$1.txt 2>../batch_jsons_out/err$2_$1
