#!/bin/bash
for i in {0..35}
do
   echo "Running input_task_ut_ang_mom_H_run1_var$i.json"
   sbatch run1-batch.sh $i
   echo "Running input_task_ut_ang_mom_H_run2_var$i.json"
   sbatch run2-batch.sh $i
   echo "Running input_task_ut_ang_mom_H_run3_var$i.json"
   sbatch run3-batch.sh $i
done