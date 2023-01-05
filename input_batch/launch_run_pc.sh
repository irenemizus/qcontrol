#!/bin/bash
for i in {1..10}
do
   let "var = $i - 1"
   echo "Running input_task_ut_ang_mom_H_run${i}_var$var.json"
   python ../newcheb.py --json_rep "input_report.json" --json_task "../batch_jsons/input_task_ut_ang_mom_H_run${i}_var$var.json" > ../batch_jsons_out/out_$i.txt 2>../batch_jsons_out/err_$i
done
