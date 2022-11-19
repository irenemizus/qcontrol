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
python ../newcheb.py --json_rep "input_report_test_dynamics_1650fs.json" --json_task "input_task_ut_ang_mom_H_2lvls_test_dynamics_1650fs.json" > ../batch_jsons_out/out_1650fs.txt 2>../batch_jsons_out/err_1650fs
