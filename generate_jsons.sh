#!/bin/bash
for i in {1..5}
do
   python newcheb.py --json_rep "input_report.json" --json_task "inp_run$i.json" --json_create
   mv input_task_ut_ang_mom_H_run$i_var* batch_jsons
done