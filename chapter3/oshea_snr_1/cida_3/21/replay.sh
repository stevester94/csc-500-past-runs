#! /bin/sh
export PYTHONPATH=/usr/local/lib/python3/dist-packages:/usr/local/lib/python3.6/dist-packages
cat << EOF | ./run.sh -
{
  "experiment_name": "OShea SNR CIDA ",
  "lr": 0.001,
  "n_epoch": 200,
  "batch_size": 128,
  "patience": 10,
  "seed": 1792,
  "device": "cuda",
  "source_snrs": [
    -18,
    -12,
    -6,
    0,
    6,
    12,
    18
  ],
  "target_snrs": [
    2,
    4,
    8,
    10,
    -20,
    14,
    16,
    -16,
    -14,
    -10,
    -8,
    -4,
    -2
  ],
  "normalize_domain": true,
  "x_net": [
    {
      "class": "Conv1d",
      "kargs": {
        "in_channels": 2,
        "out_channels": 50,
        "kernel_size": 7,
        "stride": 1,
        "padding": 0
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Conv1d",
      "kargs": {
        "in_channels": 50,
        "out_channels": 50,
        "kernel_size": 7,
        "stride": 2,
        "padding": 0
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Dropout",
      "kargs": {
        "p": 0.5
      }
    },
    {
      "class": "Flatten",
      "kargs": {}
    }
  ],
  "u_net": [
    {
      "class": "Identity",
      "kargs": {}
    }
  ],
  "merge_net": [
    {
      "class": "Linear",
      "kargs": {
        "in_features": 2901,
        "out_features": 256
      }
    }
  ],
  "class_net": [
    {
      "class": "Linear",
      "kargs": {
        "in_features": 256,
        "out_features": 256
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Dropout",
      "kargs": {
        "p": 0.5
      }
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 256,
        "out_features": 80
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 80,
        "out_features": 16
      }
    }
  ],
  "domain_net": [
    {
      "class": "Linear",
      "kargs": {
        "in_features": 256,
        "out_features": 100
      }
    },
    {
      "class": "BatchNorm1d",
      "kargs": {
        "num_features": 100
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 100,
        "out_features": 1
      }
    },
    {
      "class": "nnClamp",
      "kargs": {
        "min": -20,
        "max": 20
      }
    }
  ],
  "alpha": "sigmoid"
}
EOF