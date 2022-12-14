#!/bin/bash
for i in {0..20}
do
   for j in {1..20}
   do
       let "var = $j - 1 + $i * 20"
       echo "Running input_task_ut_ang_mom_H_run$j\_var$var.json"
       sbatch runT-batch1.sh $var $j
   done
done
