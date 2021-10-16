#! /usr/bin/env python3

from os import posix_fallocate
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.histograms import histogram

def get_experiments_from_path(start_path):
    experiment_dot_json_paths = subprocess.getoutput('find {} | grep experiment.json'.format(start_path))

    experiment_dot_json_paths = experiment_dot_json_paths.split('\n')

    experiments = []
    for p in experiment_dot_json_paths:
        with open(p) as f:
            experiments.append(json.load(f))
    
    return experiments


#################################################################
# Get the experiments, separate cida into null and sigmoid
#################################################################
cida_experiments = get_experiments_from_path("cida_1")
cnn_experiments  = get_experiments_from_path("cnn_1")

cida_alpha_null_experiments = [e for e in cida_experiments if e["parameters"]["alpha"] == "null"]
cida_alpha_sigmoid_experiments = [e for e in cida_experiments if e["parameters"]["alpha"] == "sigmoid"]


#################################################################
# Validate the data
# - Each method has the same number of experiments
# - No seeds are shared between experiments in a method
# - Ony one unique (source_snrs, target_snrs) used for all experiments
#################################################################

# Same length
assert(
    len(cnn_experiments) == len(cida_alpha_null_experiments) and
    len(cnn_experiments) == len(cida_alpha_sigmoid_experiments)
)

# No shared seeds
len([e["parameters"]["seed"] for e in cnn_experiments]) == len(set([e["parameters"]["seed"] for e in cnn_experiments]))
len([e["parameters"]["seed"] for e in cida_alpha_null_experiments]) == len(set([e["parameters"]["seed"] for e in cida_alpha_null_experiments]))
len([e["parameters"]["seed"] for e in cida_alpha_sigmoid_experiments]) == len(set([e["parameters"]["seed"] for e in cida_alpha_sigmoid_experiments]))

cnn_source_snrs = set((tuple(e["parameters"]["source_snrs"]) for e in cnn_experiments))
cida_alpha_sigmoid_source_snrs = set((tuple(e["parameters"]["source_snrs"]) for e in cida_alpha_sigmoid_experiments))
cida_alpha_null_source_snrs = set((tuple(e["parameters"]["source_snrs"]) for e in cida_alpha_null_experiments))

cnn_target_snrs = set((tuple(e["parameters"]["target_snrs"]) for e in cnn_experiments))
cida_alpha_sigmoid_target_snrs = set((tuple(e["parameters"]["target_snrs"]) for e in cida_alpha_sigmoid_experiments))
cida_alpha_null_target_snrs = set((tuple(e["parameters"]["target_snrs"]) for e in cida_alpha_null_experiments))

assert(len(cnn_source_snrs) == 1)
assert(len(cida_alpha_sigmoid_source_snrs) == 1)
assert(len(cida_alpha_null_source_snrs) == 1)
assert(len(set().union(cnn_source_snrs,cida_alpha_sigmoid_source_snrs,cida_alpha_null_source_snrs)) == 1)


assert(len(cnn_target_snrs) == 1)
assert(len(cida_alpha_sigmoid_target_snrs) == 1)
assert(len(cida_alpha_null_target_snrs) == 1)
assert(len(set().union(cnn_target_snrs, cida_alpha_sigmoid_target_snrs, cida_alpha_null_target_snrs)))

all_experiments = cida_experiments + cnn_experiments

#################################################################
# Parse relevant experiment info
#################################################################
exp = {}

# exp["cnn"] = {}
# exp["cida_alpha_null"] = {}
# exp["cida_alpha_sigmoid"] = {}

# exp["cnn"]["source_test_label_accuracy"]                = [e["results"]["source_test_label_accuracy"] for e in cnn_experiments]
# exp["cida_alpha_null"]["source_test_label_accuracy"]    = [e["results"]["source_test_label_accuracy"] for e in cida_alpha_null_experiments]
# exp["cida_alpha_sigmoid"]["source_test_label_accuracy"] = [e["results"]["source_test_label_accuracy"] for e in cida_alpha_sigmoid_experiments]

# exp["cnn"]["target_test_label_accuracy"]                = [e["results"]["target_test_label_accuracy"] for e in cnn_experiments]
# exp["cida_alpha_null"]["target_test_label_accuracy"]    = [e["results"]["target_test_label_accuracy"] for e in cida_alpha_null_experiments]
# exp["cida_alpha_sigmoid"]["target_test_label_accuracy"] = [e["results"]["target_test_label_accuracy"] for e in cida_alpha_sigmoid_experiments]

# Fuck
exp = []
exp.extend([
    {
        "seed": e["parameters"]["seed"],
        "source_test_label_accuracy": e["results"]["source_test_label_accuracy"],
        "target_test_label_accuracy": e["results"]["target_test_label_accuracy"],
        "method":"cnn",
    } for e in cnn_experiments
])

exp.extend([
    {
        "seed": e["parameters"]["seed"],
        "source_test_label_accuracy": e["results"]["source_test_label_accuracy"],
        "target_test_label_accuracy": e["results"]["target_test_label_accuracy"],
        "method":"cida_alpha_null",
    } for e in cida_alpha_null_experiments
])

exp.extend([
    {
        "seed": e["parameters"]["seed"],
        "source_test_label_accuracy": e["results"]["source_test_label_accuracy"],
        "target_test_label_accuracy": e["results"]["target_test_label_accuracy"],
        "method":"cida_alpha_sigmoid",
    } for e in cida_alpha_sigmoid_experiments
])


import pandas as pd  # This is always assumed but is included here as an introduction.

df = pd.DataFrame.from_dict(exp)

fig,ax = plt.subplots(2,3)
averages_ax = ax[0][0]
bar_source_ax = ax[0][1]
bar_target_ax = ax[0][2]
cnn_ax = ax[1][0]
cida_alpha_null_ax = ax[1][1]
cida_alpha_sigmoid_ax = ax[1][2]

# THIS WORKS
x = df[["source_test_label_accuracy","target_test_label_accuracy","method"]]
x = x.groupby("method").agg([np.mean, np.std])

averages_ax = x["target_test_label_accuracy"].plot(
    kind = "bar", y = "mean", legend = True, yerr="std",
    capsize=10, rot=10, ax=averages_ax, position=0, color="red",
    width=0.4, label="target_test_label_accuracy", alpha=0.5)
averages_ax = x["source_test_label_accuracy"].plot(
    kind = "bar", y = "mean", legend = True, yerr="std",
    capsize=10, rot=10, ax=averages_ax, position=1, color="green",
    width=0.4, label="source_test_label_accuracy", alpha=0.5)










x = df[["source_test_label_accuracy","method"]]
x = x.groupby("method")
# Results in 3 tuples: (<method name>, <dataframe of source_test_label_accuracy corresponding to that method>)

# Pos ranges from 0 to 1 and is the relative offset for each item
for group, color, pos in zip(x, ["red", "green", "blue"], [0,1,2]):
    key, group = group

    print(key)

    bar_source_ax = group.plot(
        kind = "bar", legend = True, 
        capsize=10, rot=10, ax=bar_source_ax, position=pos, color=color, 
        width=0.2, label=key, alpha=0.7)
bar_source_ax.legend([key for key,group in x])
plt.show()


# x = df[ df["method"]=="cnn"]
# df["source_test_label_accuracy"].plot.bar(y="source_test_label_accuracy", by=df["method"])
# df.groupby("method")["source_test_label_accuracy"].plot.bar()
# df["source_test_label_accuracy"].plot.bar()
# plt.show()
# print(df)



#################################################################
# Generate graphs
#################################################################

# def build_multi_bar(ax, y_list, legend_list=None, y_err_list=None, x_labels=None):
#     # Validate data
#     assert((legend_list == None) or (len(y_list) == len(legend_list))) # legend is appropriate length
#     assert(len(set([len(y) for y in y_list])) == 1) # all y same length
#     if y_err_list != None: assert(len(set([len(y) for y in y_err_list]))) == 1 # all y_errs same length

#     hist_indices = np.arange(len(y_list)+1)
#     hist_width = 0.3

#     bars = []
#     for i in range(len(y_list)):
#         print(y_list[i])
#         b = ax.bar(
#             hist_indices+i*hist_width,
#             y_list[i],
#             hist_width,
#             yerr=y_err_list[i] if y_err_list != None else None,
#             capsize=10
#         )

#         bars.append(b)

#     if legend_list is not None:
#         ax.legend(legend_list)

#     if x_labels != None:
#         ax.set_xticks(hist_indices)

#     return bars


# fig,ax = plt.subplots(2,3)
# averages_ax = ax[0][0]
# histogram_source_ax = ax[0][1]
# histogram_target_ax = ax[0][2]
# cnn_ax = ax[1][0]
# cida_alpha_null_ax = ax[1][1]
# cida_alpha_sigmoid_ax = ax[1][2]


# fig.suptitle("n={} per method".format(len(cnn_experiments)))

# build_multi_bar(
#     averages_ax,
#     y_list = [[
#         np.mean(exp["cnn"]["source_test_label_accuracy"]),
#         np.mean(exp["cida_alpha_null"]["source_test_label_accuracy"]),
#         np.mean(exp["cida_alpha_sigmoid"]["source_test_label_accuracy"]),
#     ]],
#     y_err_list = [[
#         np.std(exp["cnn"]["source_test_label_accuracy"]),
#         np.std(exp["cida_alpha_null"]["source_test_label_accuracy"]),
#         np.std(exp["cida_alpha_sigmoid"]["source_test_label_accuracy"]),
#     ]],
#     x_labels=["CNN", "CIDA Alpha null", "CIDA Alpha Sigmoid"]
# )
# averages_ax.set_title("Average and standard deviation")
# averages_ax.get_xaxis().set_visible(False)

# averages_ax.bar(
#     ["cnn", "cida, alpha null", "cida, alpha sigmoid"],
#     [
#         np.mean(exp["cnn"]["source_test_label_accuracy"]),
#         np.mean(exp["cida_alpha_null"]["source_test_label_accuracy"]),
#         np.mean(exp["cida_alpha_sigmoid"]["source_test_label_accuracy"]),
#     ],
#     yerr=[
#         np.std(exp["cnn"]["source_test_label_accuracy"]),
#         np.std(exp["cida_alpha_null"]["source_test_label_accuracy"]),
#         np.std(exp["cida_alpha_sigmoid"]["source_test_label_accuracy"]),
#     ],
#     capsize=10
# )




# Histogram of all source label accuracies
# build_multi_bar(
#     histogram_source_ax,
#     [
#         exp["cnn"]["source_test_label_accuracy"],
#         exp["cida_alpha_null"]["source_test_label_accuracy"],
#         exp["cida_alpha_sigmoid"]["source_test_label_accuracy"],
#     ],
#     ("CNN", "CIDA Alpha Null", "CIDA Alpha Sigmoid")
# )
# histogram_source_ax.get_xaxis().set_visible(False)
# histogram_source_ax.set_title("All experiments, source accuracy")



# # Histogram of all target label accuracies
# build_multi_bar(
#     histogram_target_ax,
#     [
#         exp["cnn"]["target_test_label_accuracy"],
#         exp["cida_alpha_null"]["target_test_label_accuracy"],
#         exp["cida_alpha_sigmoid"]["target_test_label_accuracy"],
#     ],
#     ("CNN", "CIDA Alpha Null", "CIDA Alpha Sigmoid")
# )
# histogram_target_ax.get_xaxis().set_visible(False)
# histogram_target_ax.set_title("All experiments, target accuracy")


# hist_indices = np.arange(len(cnn_experiments))
# cnn_hist = cnn_ax.bar(
#     hist_indices,
#     exp["cnn"]["source_test_label_accuracy"],
#     # hist_width,
# )
# cnn_ax.get_xaxis().set_visible(False)
# cnn_ax.set_ylim([0,1])
# cnn_ax.set_title("CNN")

# cida_alpha_null_hist = cida_alpha_null_ax.bar(
#     hist_indices,
#     exp["cida_alpha_null"]["source_test_label_accuracy"],
#     # hist_width,
# )
# cida_alpha_null_ax.get_xaxis().set_visible(False)
# cida_alpha_null_ax.set_ylim([0,1])
# cida_alpha_null_ax.set_title("CIDA Alpha null")

# cida_alpha_sigmoid_hist = cida_alpha_sigmoid_ax.bar(
#     hist_indices,
#     exp["cida_alpha_sigmoid"]["source_test_label_accuracy"],
#     # hist_width,
# )
# cida_alpha_sigmoid_ax.get_xaxis().set_visible(False)
# cida_alpha_sigmoid_ax.set_ylim([0,1])
# cida_alpha_sigmoid_ax.set_title("CIDA Alpha sigmoid")

