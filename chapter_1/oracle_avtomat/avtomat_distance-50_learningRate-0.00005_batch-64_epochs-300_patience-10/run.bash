#! /usr/bin/env bash
set -eou pipefail

source_dataset_base_path="/csc500-super-repo/datasets/automated_windower"
results_base_path="/csc500-super-repo/csc500-past-runs/oracle_avtomat/"

original_batch_size=100
patience=10

#Prioritize batch 128 and distances 50,56 based on epoch-10 testing
for desired_batch_size in 64 1024; do
for epochs in 300; do
for learning_rate in 0.0001 0.0005 0.00005; do
for distance in 50 56 14; do # distance 4 is busted
    experiment_name=avtomat_distance-${distance}_learningRate-${learning_rate}_batch-${desired_batch_size}_epochs-${epochs}_patience-$patience
    echo "Begin $experiment_name" | tee logs
    cat << EOF | ./avtomat_conv.py 2>&1 | tee --append logs
    {
        "experiment_name": "$experiment_name",
        "source_dataset_path": "$source_dataset_base_path/windowed_EachDevice-200k_batch-100_stride-20_distances-$distance",
        "learning_rate": $learning_rate,
        "original_batch_size": $original_batch_size,
        "desired_batch_size": $desired_batch_size,
        "epochs": $epochs,
        "patience": $patience
    }
EOF

    cp -R . $results_base_path/$experiment_name

done
done
done
done