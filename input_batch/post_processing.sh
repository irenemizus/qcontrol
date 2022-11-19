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
python ../post_processing.py $1 > ../batch_jsons_out/out_pp.txt 2>../batch_jsons_out/err_pp
