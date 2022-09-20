#!/bin/bash
for i in {0..35}
do
   echo "Running input_task_ut_ang_mom_H_var$i.json"
   sbatch run-batch.sh $i
done