#! /usr/bin/env python3

import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

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
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--cida_path', help='The path to the CIDA run')
parser.add_argument('--cnn_path', help='The path to the CNN run')
args = parser.parse_args()

cida_experiments = get_experiments_from_path(args.cida_path)
cnn_experiments  = get_experiments_from_path(args.cnn_path)

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
exp = []
exp.extend([
    {
        "seed": e["parameters"]["seed"],
        "source_val_label_accuracy": e["results"]["source_val_label_accuracy"],
        "target_val_label_accuracy": e["results"]["target_val_label_accuracy"],
        "method":"cnn",
    } for e in cnn_experiments
])

exp.extend([
    {
        "seed": e["parameters"]["seed"],
        "source_val_label_accuracy": e["results"]["source_val_label_accuracy"],
        "target_val_label_accuracy": e["results"]["target_val_label_accuracy"],
        "method":"cida_alpha_null",
    } for e in cida_alpha_null_experiments
])

exp.extend([
    {
        "seed": e["parameters"]["seed"],
        "source_val_label_accuracy": e["results"]["source_val_label_accuracy"],
        "target_val_label_accuracy": e["results"]["target_val_label_accuracy"],
        "method":"cida_alpha_sigmoid",
    } for e in cida_alpha_sigmoid_experiments
])


import pandas as pd  # This is always assumed but is included here as an introduction.

df = pd.DataFrame.from_dict(exp)

fig,ax = plt.subplots(3,3)
averages_ax = ax[0][0]
bar_source_ax = ax[0][1]
bar_target_ax = ax[0][2]
cnn_ax = ax[1][0]
cida_alpha_null_ax = ax[1][1]
cida_alpha_sigmoid_ax = ax[1][2]

fig.suptitle("n={} per method".format(len(cnn_experiments)))

#####################
# mean and std across methods
#####################
x = df[["source_val_label_accuracy","target_val_label_accuracy","method"]]
x = x.groupby("method").agg([np.mean, np.std])

averages_ax = x["target_val_label_accuracy"].plot(
    kind = "bar", y = "mean", legend = True, yerr="std",
    capsize=10, rot=10, ax=averages_ax, position=0, color="red",
    width=0.2, label="target_val_label_accuracy", alpha=0.5)
averages_ax = x["source_val_label_accuracy"].plot(
    kind = "bar", y = "mean", legend = True, yerr="std",
    capsize=10, rot=10, ax=averages_ax, position=1, color="green",
    width=0.2, label="source_val_label_accuracy", alpha=0.5)

averages_ax.set_title("Mean and StdDev by Method")
averages_ax.set_ylim([0,1])

#####################
# Source accuracy compared between methods
#####################
x = df[["source_val_label_accuracy","method"]]
x = x.groupby("method")
# Results in 3 tuples: (<method name>, <dataframe of source_val_label_accuracy corresponding to that method>)

# Pos ranges from 0 to 1 and is the relative offset for each item
for group, color, pos in zip(x, ["red", "green", "blue"], [0,1,2]):
    key, group = group

    bar_source_ax = group.plot(
        kind = "bar", legend = True, 
        capsize=10, rot=10, ax=bar_source_ax, position=pos, color=color, 
        width=0.2, label=key, alpha=0.7, stacked=False)
bar_source_ax.legend([key for key,group in x])
bar_source_ax.set_title("All experiments, source accuracy")
bar_source_ax.get_xaxis().set_visible(False)
bar_source_ax.set_ylim([0,1])
# bar_target_ax.set_xlim([-100,len(cnn_experiments)-1])


#####################
# Target accuracy compared between methods
#####################
x = df[["target_val_label_accuracy","method"]]
x = x.groupby("method")
# Results in 3 tuples: (<method name>, <dataframe of source_val_label_accuracy corresponding to that method>)

# Pos ranges from 0 to 1 and is the relative offset for each item
for group, color, pos in zip(x, ["red", "green", "blue"], [0,1,2]):
    key, group = group

    print(key)

    bar_target_ax = group.plot(
        kind = "bar", legend = True, 
        capsize=10, rot=10, ax=bar_target_ax, position=pos, color=color, 
        width=0.2, label=key, alpha=0.7)
bar_target_ax.legend([key for key,group in x])
bar_target_ax.set_title("All experiments, target accuracy")
bar_target_ax.get_xaxis().set_visible(False)
bar_target_ax.set_ylim([0,1])
bar_target_ax.set_xlim([-1,len(cnn_experiments)+1])


#####################
# Source accuracy compared between methods
#####################
x = df[df["method"] == "cnn"][["target_val_label_accuracy", "source_val_label_accuracy"]]
x.plot(kind="bar", ax=cnn_ax)
cnn_ax.set_title("CNN Accuracies")
cnn_ax.get_xaxis().set_visible(False)
cnn_ax.set_ylim([0,1])

x = df[df["method"] == "cida_alpha_sigmoid"][["target_val_label_accuracy", "source_val_label_accuracy"]]
x.plot(kind="bar", ax=cida_alpha_sigmoid_ax)
cida_alpha_sigmoid_ax.set_title("CIDA Alpha sigmoid Accuracies")
cida_alpha_sigmoid_ax.get_xaxis().set_visible(False)
cida_alpha_sigmoid_ax.set_ylim([0,1])

x = df[df["method"] == "cida_alpha_null"][["target_val_label_accuracy", "source_val_label_accuracy"]]
x.plot(kind="bar", ax=cida_alpha_null_ax)
cida_alpha_null_ax.set_title("CIDA Alpha null Accuracies")
cida_alpha_null_ax.get_xaxis().set_visible(False)
cida_alpha_null_ax.set_ylim([0,1])


# import seaborn as sns
# muh_ax = ax[0][0]

# x = df[]


# plt.show()