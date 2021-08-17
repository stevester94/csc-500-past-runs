#! /usr/bin/env python3


import re
import json
import os

experiments = []

for path in os.listdir("."):
    if not os.path.isdir(path):
        continue

    experiment_parameters_path = os.path.join(path, "experiment_parameters")
    results_path = os.path.join(path, "results", "results.txt")

    if not os.path.isfile(experiment_parameters_path):
        print(os.listdir(path))
        raise Exception("its fucked A")
    
    if not os.path.isfile(results_path):
        print(os.listdir(path))
        raise Exception("its fucked B")

    # Get the parameters as experiment, will tack on results in a second
    with open(experiment_parameters_path) as f:
        experiment = json.load(f)
    
    # Get the results



    with open(results_path) as f:
        next(f) # Skip the first line
        
        line1, line2, _ = f
        line1 = line1.rstrip().replace(":",",").replace(" ","").split(",")
        line2 = line2.rstrip().replace(":",",").replace(" ","").split(",")

        if line1[::2] != ["SourceTestLabelAcc","SourceTestLabelLoss","SourceTestDomainLoss"]: raise Exception("its fucked C")
        if line2[::2] != ["TargetTestLabelAcc","TargetTestLabelLoss","TargetTestDomainLoss"]: raise Exception("Its fucked D")

        SourceTestLabelAcc, SourceTestLabelLoss, SourceTestDomainLoss = line1[1::2]
        TargetTestLabelAcc, TargetTestLabelLoss, TargetTestDomainLoss = line2[1::2]

        experiment["SourceTestLabelAcc"]   = SourceTestLabelAcc
        experiment["SourceTestLabelLoss"]  = SourceTestLabelLoss
        experiment["SourceTestDomainLoss"] = SourceTestDomainLoss
        experiment["TargetTestLabelAcc"]   = TargetTestLabelAcc
        experiment["TargetTestLabelLoss"]  = TargetTestLabelLoss
        experiment["TargetTestDomainLoss"] = TargetTestDomainLoss

        experiments.append(experiment)


# Ok now we do all our filtering and shit here
print("window_stride","SourceTestLabelLoss","batch_size","source_distance", sep=",")

for e in experiments:
    # print(e)
    # sys.exit(1)
    print(e["window_stride"], e["SourceTestLabelLoss"], e["batch_size"], str(e["source_distance"]).replace(",", '.'), sep=",")