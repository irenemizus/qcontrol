#!/bin/bash
for i in {0..15}
do
   for j in {1..25}
   do
       let "var = $j - 1 + $i * 25"
       echo "Running input_task_ut_ang_mom_H_run$j\_var$var.json"
       sbatch runT-batch.sh $var $j
   done
done
