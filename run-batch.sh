#!/usr/bin/env bash
#python newcheb.py --json-rep "input_report.json" --json-task "input_task_ut_ang_mom_H_2lvls.json" > out.txt 2>err &
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
python math_base_tests.py