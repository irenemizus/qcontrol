#!/bin/bash
echo "Running input_task_ut_ang_mom_H_2lvls_test_dynamics_150fs.json"
sbatch run-batch_td_150fs.sh
echo "Running input_task_ut_ang_mom_H_2lvls_test_dynamics_600fs.json"
sbatch run-batch_td_600fs.sh
echo "Running input_task_ut_ang_mom_H_2lvls_test_dynamics_930fs.json"
sbatch run-batch_td_930fs.sh
echo "Running input_task_ut_ang_mom_H_2lvls_test_dynamics_1350fs.json"
sbatch run-batch_td_1350fs.sh
echo "Running input_task_ut_ang_mom_H_2lvls_test_dynamics_1650fs.json"
sbatch run-batch_td_1650fs.sh
echo "Running input_task_ut_ang_mom_H_2lvls_test_dynamics_2030fs.json"
sbatch run-batch_td_2030fs.sh