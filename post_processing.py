import collections
import copy
import os, csv
import re
import statistics
import sys

import numpy

import reporter


work_dir = sys.argv[1]
out_dir = sys.argv[2]
print(f"Input directory: {work_dir}")
print(f"Output directory: {out_dir}")

def num(s):
    try:
        return int(s)
    except ValueError:
        return numpy.float64(s)

def plot_vals_update_graph(t_list, vals_list, namem, title_plot, title_y, plot_name):
    template: str
    template_name = os.path.join(reporter.FILE_PATH, "report_templates/chart.template.html")

    xx_list = reporter.formattable_float_list()
    nt = len(t_list)
    m = reporter.mod_plot_count(nt, 1000)
    xx_list.extend([x * 1e+15 for x in t_list[0::m]])

    yyy_list_str = []
    for i in range(len(vals_list)):
        yyy_list = reporter.formattable_float_list()
        yyy_list.extend(vals_list[i][0::m])
        yyy_list_sf = str.format(f"{yyy_list:.4e}")
        yyy_list_str.append(
            '{' + f" \"t\": \"{namem[i]}\", \"values\": {yyy_list_sf}, \"pointRadius\": 0 " + '}'
        )
    substs = {
        "{{TITLE}}":    "\"" + title_plot + "\"",
        "{{X_TITLE}}":  "\"time, fs\"",
        "{{Y_TITLE}}":  "\"" + title_y + "\"",
        "{{T_TITLE}}":  "\"\"",
        "{{XX_LIST}}":  str.format(f"{xx_list:.2f}"),
        "{{YYY_LIST}}": "[ " + ", ".join(yyy_list_str) + " ]"
    }

    inst = reporter.templateSubst(template_name, substs)

    with open(os.path.join(out_dir, plot_name), "w") as f:
        f.write(inst)

class batch_result:
    iter: int
    goal_close: numpy.float64
    F_sm: numpy.float64
    E_int: numpy.float64
    J: numpy.float64

    def __init__(self, iter, goal_close, F_sm, E_int, J):
        self.iter = iter
        self.goal_close = goal_close
        self.F_sm = F_sm
        self.E_int = E_int
        self.J = J

    def __str__(self):
        return "[ #" + str(self.iter) + ", gc: " + str(self.goal_close) + ", F_sm: " + str(self.F_sm) + \
               ", E_int: " + str(self.E_int) + ", J: " + str(self.J) + " ]"


root = work_dir
run_dirs = [f.path for f in os.scandir(root) if f.is_dir()]
print(run_dirs)

prog_nb = re.compile("nb:\t+(\d+)$")
prog_it = re.compile("iter_mid_2:\t+(\d+)$")
prog_eps = re.compile("epsilon:\t+(.+)$")

nb = 0
iter_mid_2 = 0
epsilon = 0.0

iter_l = 0
Fsm_l = 0.0
gc_l = 0.0
E_int_l = 0.0
J_l = 0.0
looking_for_min = True

# Reading all the data from tab_iter.csv, together with run_id and T value
runs = dict()
for run_dir in run_dirs:
    time_dirs = [f.path for f in os.scandir(run_dir) if f.is_dir()]
    run_id = os.path.split(run_dir)[-1]

    times = dict()
    for time_dir in time_dirs:
        time_val = numpy.float64(os.path.split(time_dir)[-1].split('T=')[-1])
        print(time_dir)

        # First, we need to get a few parameters from the file "table_inp_-1.txt"
        with open(os.path.join(time_dir, "table_inp_-1.txt"), "r") as finp:
            lines = finp.readlines()
        for l in lines:
            res_nb = prog_nb.match(l)
            res_it = prog_it.match(l)
            res_eps = prog_eps.match(l)

            if res_nb: nb = int(res_nb.group(1))
            if res_it: iter_mid_2 = int(res_it.group(1))
            if res_eps: epsilon = numpy.float64(res_eps.group(1))

        data = None
        with open(os.path.join(time_dir, "tab_iter.csv"), "r") as f:
            try:
                reader = csv.reader(f, delimiter=' ')
                data = list(reader)
            except csv.Error:
                print (f"An error occured while reading the file 'tab_iter.csv' in folder {time_dir}")

            #import pdb; pdb.set_trace()
            #print(data)
            if data is not None:
                for i in range(len(data)):
                    data[i] = [x for x in data[i] if x != '']
                    data[i] = batch_result(int(data[i][0]), numpy.float64(data[i][1]), numpy.float64(data[i][2]), numpy.float64(data[i][3]), numpy.float64(data[i][4]))
        if data is not None:
            if nb and iter_mid_2 and epsilon:
                iter_l = data[-1].iter
                Fsm_l = data[-1].F_sm
                gc_l = data[-1].goal_close
                E_int_l = data[-1].E_int
                J_l = data[-1].J
                if (iter_l <= iter_mid_2) and (Fsm_l >= -nb * nb + epsilon):
                    continue
                elif (iter_l <= iter_mid_2) and (Fsm_l < -nb * nb + epsilon):
                    looking_for_min = False
                    times[time_val] = (data, looking_for_min)
                elif iter_l > iter_mid_2:
                    looking_for_min = True
                    times[time_val] = (data, looking_for_min)
            else:
                times[time_val] = (data, looking_for_min)

    runs[run_id] = times

# Searching for the minimum value of F_sm together with corresponding E_int and J values during the iterative procedure
# for each run_id and each T value
runs_min = dict()
for r in runs:
    runs_min[r] = dict()
    run = runs[r]

    for t in run:
        time_min = None
        F_sm_min = 0.0
        iter_min = 0
        time = run[t][0]
        looking_for_min_tr = run[t][1]

        if looking_for_min_tr:
            for tt in time:
                if tt.F_sm < F_sm_min:
                    F_sm_min = tt.F_sm
                    iter_min = tt.iter
                    time_min = copy.deepcopy(tt)
            if iter_min <= iter_mid_2:
                time_min = None
                F_sm_min = 0.0
        else:
            F_sm_min = time[-1].F_sm
            time_min = time[-1]
        #print(r, F_sm_min)

        if time_min is not None:
            runs_min[r][t] = time_min
        pass

# Writing the tables with minimum values of F_sm during the iterative procedures for each run_id and each T value into a set of txt files
glob_tabs_path = os.path.join(out_dir, "glob_tabs")
os.makedirs(glob_tabs_path, exist_ok=True)

for r in runs_min:
    with open(os.path.join(glob_tabs_path, "glob_" + r + ".txt"), "w") as fout:
        run = runs_min[r]

        for t in run:
            time = run[t]

            it = time.iter
            F_sm = time.F_sm
            fout.write(f"{t}\t{it}\t{F_sm}\n")
            fout.flush()

# Inverting the runs_min dictionary
runs_inv = dict()

for r in runs_min:
    run = runs_min[r]
    for t in run:
        time = run[t]

        if t not in runs_inv:
            runs_inv[t] = dict()

        if r not in runs_inv[t]:
            runs_inv[t][r] = []

        runs_inv[t][r] = time

runs_inv_sort = collections.OrderedDict(sorted(runs_inv.items()))

# Getting the stats for all the runs
runs_min_min = dict()
runs_min_avg = dict()
runs_min_med = dict()

runs_min_min_E = dict()
runs_min_min_J = dict()

min_min_run = []
F_sm_min_full_data = dict()
for t in runs_inv_sort:
    time = runs_inv_sort[t]

    times_list = []
    runs_list = []
    for r in time:
        times_list.append(time[r].F_sm)
        runs_list.append((r, time[r]))

    # Looking for the min F_sm in runs_list
    F_sm_min = runs_list[0]
    F_sm_min_indx = 0
    for i in range(len(runs_list)):
        new_F_sm = runs_list[i]
        if new_F_sm[1].F_sm < F_sm_min[1].F_sm:
            F_sm_min = new_F_sm
            F_sm_min_indx = i

    F_sm_min_full_data[t] = F_sm_min

    runs_min_min[t] = min(times_list)
    runs_min_avg[t] = statistics.mean(times_list)
    runs_min_med[t] = statistics.median(times_list)

# Plotting the results to txt format

# Writing the total table with minimum values of F_sm during the iterative procedure for each run_id for each T value
with open(os.path.join(out_dir, "glob.txt"), "w") as f_gl:
    for t in runs_inv_sort:
        tfs = t * 1e+15
        f_gl.write("T = {:.1f} fs\n".format(tfs))
        time = runs_inv_sort[t]
        for r in time:
            it = time[r].iter
            F_sm = time[r].F_sm
            f_gl.write(f"{r}\t{it}\t{F_sm}\n")
            f_gl.flush()
        f_gl.write("\n")

# Writing the table with minimum values of F_sm among all the runs as a function of T value
with open(os.path.join(out_dir, "glob_F_graph.txt"), "w") as f_fgr:
    for t in runs_inv_sort:
        tfs = numpy.float64(t) * 1e+15
        min = runs_min_min[t]
        avg = runs_min_avg[t]
        med = runs_min_med[t]
        f_fgr.write("{:.1f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(tfs, min, avg, med))
        f_fgr.flush()

# Writing the table with integral energies of optimal laser field corresponding to the minimum values of F_sm among
# all the runs as a function of T value
E_int_dict = dict()
with open(os.path.join(out_dir, "glob_E_int_graph.txt"), "w") as f_eigr:
    for t in F_sm_min_full_data:
        tfs = numpy.float64(t) * 1e+15
        time = F_sm_min_full_data[t]
        E2_int = time[1].E_int
        E_int_dict[t] = E2_int

        f_eigr.write("{:.4f}\t{:.4f}\n".format(tfs, E2_int))
        f_eigr.flush()

# Plotting the results to html format

# Plotting the graph for extreme F values
t_list = []
min_list = []
avg_list = []
med_list = []

for t in runs_min_min:
    t_list.append(t)
    min_list.append(runs_min_min[t])
    avg_list.append(runs_min_avg[t])
    med_list.append(runs_min_med[t])

namef = ["F_min", "F_avg", "F_med"]
vals_list = [min_list, avg_list, med_list]

# Updating the graph for extreme F values
plot_vals_update_graph(t_list, vals_list, namef,
                              "Statistics for the extreme values of operator F_sm", "",
                              "glob_F_graph.html")

# Plotting the graph for integral energy of optimal laser field
E_int_list = []
for t in t_list:
    if t in E_int_dict:
        E_int_list.append(E_int_dict[t])
    else:
        E_int_list.append(0.0)

namee = ["E_int"]
vals_list = [E_int_list]

# Updating the graph for integral energy of optimal laser field
plot_vals_update_graph(t_list, vals_list, namee,
                              "Integral energy of the optimal laser field", "",
                              "glob_E_int_graph.html")
