#! /usr/bin/env python3

import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

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
exp = []
exp.extend([
    {
        "seed": e["parameters"]["seed"],
        "source_test_label_accuracy": e["results"]["source_test_label_accuracy"],
        "target_test_label_accuracy": e["results"]["target_test_label_accuracy"],
        "source_val_label_accuracy": e["results"]["source_val_label_accuracy"],
        "target_val_label_accuracy": e["results"]["target_val_label_accuracy"],
        "method":"cnn",
    } for e in cnn_experiments
])

exp.extend([
    {
        "seed": e["parameters"]["seed"],
        "source_test_label_accuracy": e["results"]["source_test_label_accuracy"],
        "target_test_label_accuracy": e["results"]["target_test_label_accuracy"],
        "source_val_label_accuracy": e["results"]["source_val_label_accuracy"],
        "target_val_label_accuracy": e["results"]["target_val_label_accuracy"],
        "method":"cida_alpha_null",
    } for e in cida_alpha_null_experiments
])

exp.extend([
    {
        "seed": e["parameters"]["seed"],
        "source_test_label_accuracy": e["results"]["source_test_label_accuracy"],
        "target_test_label_accuracy": e["results"]["target_test_label_accuracy"],
        "source_val_label_accuracy": e["results"]["source_val_label_accuracy"],
        "target_val_label_accuracy": e["results"]["target_val_label_accuracy"],
        "method":"cida_alpha_sigmoid",
    } for e in cida_alpha_sigmoid_experiments
])


import pandas as pd  # This is always assumed but is included here as an introduction.

df = pd.DataFrame.from_dict(exp)

print(df.groupby("method").aggregate("mean"))