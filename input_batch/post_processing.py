import collections
import copy
import os, csv
import statistics
import sys
import reporter


work_dir = sys.argv[1]

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

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

    with open(plot_name, "w") as f:
        f.write(inst)

class batch_result:
    iter: int
    goal_close: float
    F_sm: float

    def __init__(self, iter, goal_close, F_sm):
        self.iter = iter
        self.goal_close = goal_close
        self.F_sm = F_sm

    def __str__(self):
        return "[ #" + str(self.iter) + ", gc: " + str(self.goal_close) + ", F_sm: " + str(self.F_sm) + " ]"


root = work_dir
run_dirs = [f.path for f in os.scandir(root) if f.is_dir()]
print(run_dirs)

# Reading all the data from tab_iter.csv, together with run_id and T value
runs = dict()
for run_dir in run_dirs:
    time_dirs = [f.path for f in os.scandir(run_dir) if f.is_dir()]
    run_id = os.path.split(run_dir)[-1]

    times = dict()
    for time_dir in time_dirs:
        time_val = float(os.path.split(time_dir)[-1].split('T=')[-1])

        print(time_dir)
        with open(os.path.join(time_dir, "tab_iter.csv")) as f:
            reader = csv.reader(f, delimiter=' ')
            data = list(reader)

            for i in range(len(data)):
                data[i] = [x for x in data[i] if x != '']
                data[i] = batch_result(int(data[i][0]), float(data[i][1]), float(data[i][2]))

        times[time_val] = data

    runs[run_id] = times

# Searching fot the minimum value of F_sm during the iterative procedure for each run_id and each T value
runs_min = dict()
for r in runs:
    runs_min[r] = dict()
    run = runs[r]

    for t in run:
        time_min = None
        F_sm_min = 0.0
        time = run[t]

        for tt in time:
            if tt.F_sm < F_sm_min:
                F_sm_min = tt.F_sm
                time_min = copy.deepcopy(tt)
        if time_min is None:
            time_min = batch_result(0, 0.0, 0.0)
        runs_min[r][t] = time_min
        pass

# Writing the tables with minimum values of F_sm during the iterative procedures for each run_id and each T value into a set of txt files
if not os.path.exists("glob_tabs"):
    os.makedirs("glob_tabs", exist_ok=True)

for r in runs_min:
    with open(os.path.join("glob_tabs", "glob_" + r + ".txt"), "w") as fout:
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
with open("glob.txt", "w") as f_gl:
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
with open("glob_F_graph.txt", "w") as f_fgr:
    for t in runs_inv_sort:
        tfs = float(t) * 1e+15
        min = runs_min_min[t]
        avg = runs_min_avg[t]
        med = runs_min_med[t]
        f_fgr.write("{:.1f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(tfs, min, avg, med))
        f_fgr.flush()

# Writing the table with integral energies of optimal laser field corresponding to the minimum values of F_sm among all the runs as a function of T value
E_int_dict = dict()
with open("glob_E_graph.txt", "w") as f_egr:
    for tmin in F_sm_min_full_data:
        fmin = F_sm_min_full_data[tmin]

        rmin = fmin[0]
        imin = fmin[1].iter

        E_int = 0.0
        t_prev = -1.0
        Efile_path = os.path.join(rmin, "T=" + str(tmin), "tab_iter_E.csv")
        ifExists = os.path.exists(Efile_path)
        if ifExists:
            for line_E in open(Efile_path, "r"):
                ll = line_E.strip().split(" ")
                step_E = int(ll[0])
                t = float(ll[1])
                if step_E == imin:
                    if t_prev < 0: t_prev = t
                    E_cur = float(ll[-1])
                    E_int += E_cur * (t - t_prev)
                    t_prev = t
            tminfs = tmin * 1e+15
            E_int_dict[tmin] = E_int
            f_egr.write("{:.4f}\t{:.4f}\n".format(tminfs, E_int))
            f_egr.flush()

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
                              "glob_E_graph.html")