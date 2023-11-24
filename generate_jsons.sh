#!/bin/bash
python newcheb.py --json_rep "input_batch/input_report.json" --json_task "input_batch/inp_Jx_13400-26Kfs_25runs.json" --json_create
mv input_task_ut_ang_mom_H_run* batch_jsons
