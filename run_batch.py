import os
import threading

if not os.path.exists("batch_runs_inp/rand_ampls/generated"):
    os.mkdir("batch_runs_inp/rand_ampls/generated")

def generate_report(template_filename, report_filename, out_path):
    with open(template_filename, "r") as tf:
        with open(report_filename, "w") as rf:
            for line in tf:
                line = line.replace("{OUT_PATH}", out_path)
                rf.write(line)

global_mutex = threading.Lock()

def thread_run(r, suff):
    with global_mutex:
        print(f"Thread #{r}{suff} started")
    generate_report("batch_runs_inp/rand_ampls/input_report.json.template",
                    f"batch_runs_inp/rand_ampls/generated/input_report_gen{suff}_{r}.json", "output_ut_" + str(r) + suff)
    os.system(
        f'python newcheb.py --json_task "batch_runs_inp/rand_ampls/input_task{suff}.json" --json_rep "batch_runs_inp/rand_ampls/generated/input_report_gen{suff}_{r}.json" > batch_runs_inp/rand_ampls/generated/out_{r}{suff}.txt')
    with global_mutex:
        print(f"Thread #{r} finished")

count = 10
threads_list = []
for r in range(count):
    new_thread = threading.Thread(target=thread_run, args=[ r, "" ])
    new_thread_p4 = threading.Thread(target=thread_run, args=[ r, "_p4" ])
    with global_mutex:
        print(f"Starting thread #{r}")
    with global_mutex:
        print(f"Starting thread #{r}_p4")

    new_thread.start()
    new_thread_p4.start()
    threads_list.append(new_thread)
    threads_list.append(new_thread_p4)

for r in range(2 * count):
    if r % 2:
        with global_mutex:
            print(f"Waiting for thread #{r // 2}")
    else:
        with global_mutex:
            print(f"Waiting for thread #{r // 2}_p4")
    threads_list[r].join()

with open("batch_runs_inp/rand_ampls/generated/files_success.txt", "w") as fs:
    for r in range(count):
        with open(os.path.join("output_ut_" + str(r), "tab_iter.csv"), "r") as fo:
            lines = fo.readlines()
            for l in lines:
                if float(l.split()[-1].strip()) < -3.99:
                    fs.write(f"output_ut_{r},    Iteration = {l.split()[0].strip()}\n")
        with open(os.path.join("output_ut_" + str(r) + "_p4", "tab_iter.csv"), "r") as fop4:
            lines = fop4.readlines()
            for l in lines:
                if float(l.split()[-1].strip()) < -3.99:
                    fs.write(f"output_ut_{r}_p4,    Iteration = {l.split()[0].strip()}\n")
