#!/bin/bash
for i in {1..10}
do
   echo "Running input_task_ut_ang_mom_H_run$i.json"
   sbatch run-batch.sh $i
done
