#! /usr/bin/env bash

regex="avtomat_distance-([0-9]+)_learningRate-(0\.[0-9]+)_batch-([0-9]+)_epochs-([0-9]+)_patience-([0-9]+)"

dirs=$(find . -mindepth 1 -maxdepth 1 -type d)

echo NAME,DISTANCE,LEARNING_RATE,BATCH,EPOCHS,PATIENCE,TEST_LOSS,TEST_ACC,VAL_LOSS,VAL_ACC,TOTAL_TIME_SECS

for d in $dirs; do
    name=$(cat $d/RESULTS | grep "Experiment name" | awk '{print $3}')
    test_loss=$(cat $d/RESULTS | grep "test loss" | awk '{print $2}' | tr -d 'loss:,')
    test_acc=$(cat $d/RESULTS | grep "test loss" | awk '{print $4}' | tr -d 'ac:,')
    val_loss=$(cat $d/RESULTS | grep "val loss" | awk '{print $2}' | tr -d 'loss:,')
    val_acc=$(cat $d/RESULTS | grep "val loss" | awk '{print $4}' | tr -d 'ac:,')
    total_time_secs=$(cat $d/RESULTS | grep "total time seconds" | awk '{print $4}')


    if [[ $name =~ $regex ]]; then
        distance="${BASH_REMATCH[1]}"
        learning_rate="${BASH_REMATCH[2]}"
        batch="${BASH_REMATCH[3]}"
        epochs="${BASH_REMATCH[4]}"
        patience="${BASH_REMATCH[5]}"
    else
        echo "Failed to match regex"
        exit 1
    fi

    echo $name,$distance,$learning_rate,$batch,$epochs,$patience,$test_loss,$test_acc,$val_loss,$val_acc,$total_time_secs

done
