#! /usr/bin/python3

# This is just a toy to test out basic keras usage. We are training an identity function


import sys, os

from tensorflow.python.ops.gen_math_ops import floor

from steves_utils.graphing import plot_confusion_matrix, plot_loss_curve, save_confusion_matrix, save_loss_curve
from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
from steves_utils.ORACLE.utils import ALL_DISTANCES_FEET, ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS
import steves_utils.utils

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import math
import random

import time

EXPERIMENT_NAME = "512_batch_200kXdev_samples_windowing_75epochs_learnrate_0.001_distanceEQ-2"

# Setting the seed is vital for reproducibility
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def get_shuffled_and_windowed_from_pregen_ds():
    from steves_utils.ORACLE.windowed_shuffled_dataset_accessor import Windowed_Shuffled_Dataset_Factory
    from steves_utils import utils

    # Batch size is baked into the dataset
    path = os.path.join(utils.get_datasets_base_path(), "windowed_200k-each-devices_batch-100")
    print(utils.get_datasets_base_path())
    print(path)
    datasets = Windowed_Shuffled_Dataset_Factory(path)

    ORIGINAL_BATCH_SIZE=100
    DESIRED_BATCH_SIZE=512

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]

    train_ds = train_ds.unbatch()
    val_ds  = val_ds.unbatch()
    test_ds = test_ds.unbatch()

    train_ds = train_ds.filter(lambda x: x["distance_feet"] == 2)
    val_ds = val_ds.filter(lambda x: x["distance_feet"] == 2)
    test_ds = test_ds.filter(lambda x: x["distance_feet"] == 2)

    train_ds = train_ds.batch(DESIRED_BATCH_SIZE)
    val_ds  = val_ds.batch(DESIRED_BATCH_SIZE)
    test_ds = test_ds.batch(DESIRED_BATCH_SIZE)

    train_ds = train_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    train_ds = train_ds.unbatch()
    # val_ds  = val_ds.unbatch()
    # test_ds = test_ds.unbatch()

    train_ds = train_ds.shuffle(100 * ORIGINAL_BATCH_SIZE, reshuffle_each_iteration=True)
    
    train_ds = train_ds.batch(DESIRED_BATCH_SIZE)
    # val_ds  = val_ds.batch(DESIRED_BATCH_SIZE)
    # test_ds = test_ds.batch(DESIRED_BATCH_SIZE)

    train_ds = train_ds.prefetch(100)
    val_ds   = val_ds.prefetch(100)
    test_ds  = test_ds.prefetch(100)

    return train_ds, val_ds, test_ds


def get_all_shuffled_windowed():
    global RANGE
    from steves_utils.ORACLE.shuffled_dataset_accessor import Shuffled_Dataset_Factory
    from steves_utils import utils

    DATASET_BATCH_SIZE = 100
    BATCH = 256
    chunk_size = 4 * ORIGINAL_PAPER_SAMPLES_PER_CHUNK
    STRIDE_SIZE=1

    NUM_REPEATS= math.floor((chunk_size - ORIGINAL_PAPER_SAMPLES_PER_CHUNK)/STRIDE_SIZE) + 1

    path = os.path.join(utils.get_datasets_base_path(), "all_shuffled_chunk-512", "output")
    print(utils.get_datasets_base_path())
    print(path)
    datasets = Shuffled_Dataset_Factory(
        path, train_val_test_splits=(0.6, 0.2, 0.2), reshuffle_train_each_iteration=False
    )

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]

    train_ds = train_ds.unbatch().take(200000 * len(ALL_SERIAL_NUMBERS))
    val_ds = val_ds.unbatch().take(10000 * len(ALL_SERIAL_NUMBERS))
    test_ds = test_ds.unbatch().take(50000 * len(ALL_SERIAL_NUMBERS))

    train_ds = train_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    train_ds = train_ds.map(
        lambda x, y:
        (
            tf.transpose(
                tf.signal.frame(x, ORIGINAL_PAPER_SAMPLES_PER_CHUNK, STRIDE_SIZE),
                [1,0,2]
            ),
            tf.repeat(tf.reshape(y, (1,RANGE)), repeats=NUM_REPEATS, axis=0)
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    # We aren't really windowing the val and test data, we are just splitting them into 128 sample chunks so that they are
    # the same shape as the train data
    val_ds = val_ds.map(
        lambda x, y:
        (
            tf.transpose(
                # See, stride == length, meaning we are just splitting the chunks, not really windowing
                tf.signal.frame(x, ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ORIGINAL_PAPER_SAMPLES_PER_CHUNK),
                [1,0,2]
            ),
            tf.repeat(tf.reshape(y, (1,RANGE)), repeats=math.floor(chunk_size/ORIGINAL_PAPER_SAMPLES_PER_CHUNK), axis=0)
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x, y:
        (
            tf.transpose(
                # See, stride == length, meaning we are just splitting the chunks, not really windowing
                tf.signal.frame(x, ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ORIGINAL_PAPER_SAMPLES_PER_CHUNK),
                [1,0,2]
            ),
            tf.repeat(tf.reshape(y, (1,RANGE)), repeats=math.floor(chunk_size/ORIGINAL_PAPER_SAMPLES_PER_CHUNK), axis=0)
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )


    train_ds = train_ds.unbatch().take(200000*len(ALL_SERIAL_NUMBERS))
    val_ds = val_ds.unbatch().take(10000*len(ALL_SERIAL_NUMBERS))
    test_ds = test_ds.unbatch().take(50000*len(ALL_SERIAL_NUMBERS))

    train_ds = train_ds.shuffle(DATASET_BATCH_SIZE*NUM_REPEATS*3, reshuffle_each_iteration=True)

    train_ds = train_ds.batch(BATCH)
    val_ds   = val_ds.batch(BATCH)
    test_ds  = test_ds.batch(BATCH)

    train_ds = train_ds.prefetch(100)
    val_ds   = val_ds.prefetch(100)
    test_ds  = test_ds.prefetch(100)



    return train_ds, val_ds, test_ds


def get_all_shuffled():
    global RANGE
    from steves_utils.ORACLE.shuffled_dataset_accessor import Shuffled_Dataset_Factory
    from steves_utils import utils

    BATCH = 256

    path = os.path.join(utils.get_datasets_base_path(), "all_shuffled", "output")
    print(utils.get_datasets_base_path())
    print(path)
    datasets = Shuffled_Dataset_Factory(
        path, train_val_test_splits=(0.6, 0.2, 0.2), reshuffle_train_each_iteration=False
    )

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]    

    train_ds = train_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    train_ds = train_ds.unbatch().take(200000 * len(ALL_SERIAL_NUMBERS)).batch(BATCH)
    val_ds = val_ds.unbatch().take(10000 * len(ALL_SERIAL_NUMBERS)).batch(BATCH)
    test_ds = test_ds.unbatch().take(50000 * len(ALL_SERIAL_NUMBERS)).batch(BATCH)

    return train_ds, val_ds, test_ds

# This works
def get_limited_oracle():
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = (0.6, 0.2, 0.2)
    BATCH=1000
    RANGE   = len(ALL_SERIAL_NUMBERS)

    
    ds, cardinality = Simple_ORACLE_Dataset_Factory(
        ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
        runs_to_get=[1],
        distances_to_get=[8],
        serial_numbers_to_get=ALL_SERIAL_NUMBERS[:3]
    )

    print("Total Examples:", cardinality)
    print("That's {}GB of data (at least)".format( cardinality * ORIGINAL_PAPER_SAMPLES_PER_CHUNK * 2 * 8 / 1024 / 1024 / 1024))

    num_train = int(cardinality * TRAIN_SPLIT)
    num_val = int(cardinality * VAL_SPLIT)
    num_test = int(cardinality * TEST_SPLIT)

    ds = ds.shuffle(cardinality)

    train_ds = ds.take(num_train)
    val_ds = ds.skip(num_train).take(num_val)
    test_ds = ds.skip(num_train+num_val).take(num_test)

    train_ds = train_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    train_ds = train_ds.batch(BATCH)
    val_ds   = val_ds.batch(BATCH)
    test_ds  = test_ds.batch(BATCH)

    return train_ds, val_ds, test_ds

def get_less_limited_oracle():
    """test loss: 0.05208379030227661 , test acc: 0.16599488258361816"""
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = (0.6, 0.2, 0.2)
    BATCH=500
    RANGE   = len(ALL_SERIAL_NUMBERS)

    
    ds, cardinality = Simple_ORACLE_Dataset_Factory(
        ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
        runs_to_get=[1],
        distances_to_get=ALL_DISTANCES_FEET[:1],
        serial_numbers_to_get=ALL_SERIAL_NUMBERS[:6]
    )

    print("Total Examples:", cardinality)
    print("That's {}GB of data (at least)".format( cardinality * ORIGINAL_PAPER_SAMPLES_PER_CHUNK * 2 * 8 / 1024 / 1024 / 1024))
    input("Pres Enter to continue")
    num_train = int(cardinality * TRAIN_SPLIT)
    num_val = int(cardinality * VAL_SPLIT)
    num_test = int(cardinality * TEST_SPLIT)

    ds = ds.shuffle(cardinality)
    ds = ds.cache(os.path.join(steves_utils.utils.get_datasets_base_path(), "caches", "less_limited_oracle"))

    # # Prime the cache
    # for e in ds.batch(1000):
    #     pass

    # print("Buffer primed. Comment this out next time")
    # sys.exit(1)

    train_ds = ds.take(num_train)
    val_ds = ds.skip(num_train).take(num_val)
    test_ds = ds.skip(num_train+num_val).take(num_test)

    train_ds = train_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )
    
    train_ds = train_ds.batch(BATCH)
    val_ds   = val_ds.batch(BATCH)
    test_ds  = test_ds.batch(BATCH)

    train_ds = train_ds.prefetch(100)
    val_ds   = val_ds.prefetch(100)
    test_ds  = test_ds.prefetch(100)

    return train_ds, val_ds, test_ds

def get_windowed_less_limited_oracle():
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = (0.6, 0.2, 0.2)
    BATCH=500
    RANGE   = len(ALL_SERIAL_NUMBERS)

    chunk_size = 4 * ORIGINAL_PAPER_SAMPLES_PER_CHUNK
    STRIDE_SIZE=1

    NUM_REPEATS= math.floor((chunk_size - ORIGINAL_PAPER_SAMPLES_PER_CHUNK)/STRIDE_SIZE) + 1

    ds, cardinality = Simple_ORACLE_Dataset_Factory(
        chunk_size, 
        runs_to_get=[1],
        distances_to_get=ALL_DISTANCES_FEET[:1],
        serial_numbers_to_get=ALL_SERIAL_NUMBERS[:6]
    )

    print("Total Examples:", cardinality)
    print("That's {}GB of data (at least)".format( cardinality * chunk_size * 2 * 8 / 1024 / 1024 / 1024))
    # input("Pres Enter to continue")
    num_train = int(cardinality * TRAIN_SPLIT)
    num_val = int(cardinality * VAL_SPLIT)
    num_test = int(cardinality * TEST_SPLIT)

    ds = ds.shuffle(cardinality)
    ds = ds.cache(os.path.join(steves_utils.utils.get_datasets_base_path(), "caches", "windowed_less_limited_oracle"))

    # Prime the cache

    # for e in ds.batch(1000):
    #     pass

    # print("Buffer primed. Comment this out next time")
    # sys.exit(1)

    train_ds = ds.take(num_train)
    val_ds = ds.skip(num_train).take(num_val)
    test_ds = ds.skip(num_train+num_val).take(num_test)

    train_ds = train_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    train_ds = train_ds.map(
        lambda x, y:
        (
            tf.transpose(
                tf.signal.frame(x, ORIGINAL_PAPER_SAMPLES_PER_CHUNK, STRIDE_SIZE), # Somehow we get 9 frames from this
                [1,0,2]
            ),
            tf.repeat(tf.reshape(y, (1,RANGE)), repeats=NUM_REPEATS, axis=0) # Repeat our one hot tensor 9 times
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x, y:
        (
            tf.transpose(
                tf.signal.frame(x, ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ORIGINAL_PAPER_SAMPLES_PER_CHUNK), # Somehow we get 9 frames from this
                [1,0,2]
            ),
            tf.repeat(tf.reshape(y, (1,RANGE)), repeats=math.floor(chunk_size/ORIGINAL_PAPER_SAMPLES_PER_CHUNK), axis=0) # Repeat our one hot tensor 9 times
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x, y:
        (
            tf.transpose(
                tf.signal.frame(x, ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ORIGINAL_PAPER_SAMPLES_PER_CHUNK), # Somehow we get 9 frames from this
                [1,0,2]
            ),
            tf.repeat(tf.reshape(y, (1,RANGE)), repeats=math.floor(chunk_size/ORIGINAL_PAPER_SAMPLES_PER_CHUNK), axis=0) # Repeat our one hot tensor 9 times
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    # for e in train_ds:
    #     print(e)

    # for e in test_ds:
    #     print(e)

    # sys.exit(1)


    train_ds = train_ds.unbatch()
    val_ds = val_ds.unbatch()
    test_ds = test_ds.unbatch()

    train_ds = train_ds.shuffle(BATCH*NUM_REPEATS*3)
    val_ds   = val_ds.shuffle(BATCH*NUM_REPEATS*3)
    test_ds  = test_ds.shuffle(BATCH*NUM_REPEATS*3)

    train_ds = train_ds.batch(BATCH)
    val_ds   = val_ds.batch(BATCH)
    test_ds  = test_ds.batch(BATCH)

    train_ds = train_ds.prefetch(100)
    val_ds   = val_ds.prefetch(100)
    test_ds  = test_ds.prefetch(100)

    return train_ds, val_ds, test_ds

def get_windowed_foxtrot_shuffled():
    from steves_utils.ORACLE.shuffled_dataset_accessor import Shuffled_Dataset_Factory
    from steves_utils import utils

    path = os.path.join(utils.get_datasets_base_path(), "foxtrot", "output")
    datasets = Shuffled_Dataset_Factory(path, train_val_test_splits=(0.6, 0.2, 0.2))

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]    

    # count = 0
    # for e in train_ds.concatenate(val_ds).concatenate(test_ds):
    #     count += e["IQ"].shape[0]
    # print(count)
    # sys.exit(1)

    train_ds = train_ds.unbatch()
    val_ds = val_ds.unbatch()
    test_ds = test_ds.unbatch()

    

    # Chunk size and batch is determined by the shuffled dataset
    chunk_size = 4 * ORIGINAL_PAPER_SAMPLES_PER_CHUNK
    STRIDE_SIZE=1
    BATCH=1000
    REBATCH=500

    NUM_REPEATS= math.floor((chunk_size - ORIGINAL_PAPER_SAMPLES_PER_CHUNK)/STRIDE_SIZE) + 1

    # print(RANGE)
    # sys.exit(1)

    # serial_number_id ranges from [0,15]

    # train_ds = train_ds.filter(lambda x: x["serial_number_id"] < 13 or x["serial_number_id"] > 13)
    # val_ds = val_ds.filter(lambda x: x["serial_number_id"] < 13 or x["serial_number_id"] > 13)
    # test_ds = test_ds.filter(lambda x: x["serial_number_id"] < 13 or x["serial_number_id"] > 13)

    # train_ds = train_ds.filter(lambda x: x["serial_number_id"] !=  13)
    # val_ds = val_ds.filter(lambda x: x["serial_number_id"] !=  13)
    # test_ds = test_ds.filter(lambda x: x["serial_number_id"]  != 13)

    # train_ds = train_ds.filter(lambda x: x["serial_number_id"] < 15)
    # val_ds = val_ds.filter(lambda x: x["serial_number_id"]     < 15)
    # test_ds = test_ds.filter(lambda x: x["serial_number_id"]   < 15)

    # val_ds = val_ds.filter(lambda x: x["serial_number_id"] in target_serials)
    # test_ds = test_ds.filter(lambda x: x["serial_number_id"] in target_serials)


    train_ds = train_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x: (x["IQ"],tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )


    

    train_ds = train_ds.map(
        lambda x, y:
        (
            tf.transpose(
                tf.signal.frame(x, ORIGINAL_PAPER_SAMPLES_PER_CHUNK, STRIDE_SIZE), # Somehow we get 9 frames from this
                [1,0,2]
            ),
            tf.repeat(tf.reshape(y, (1,RANGE)), repeats=NUM_REPEATS, axis=0) # Repeat our one hot tensor 9 times
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    val_ds = val_ds.map(
        lambda x, y:
        (
            tf.transpose(
                tf.signal.frame(x, ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ORIGINAL_PAPER_SAMPLES_PER_CHUNK), # Somehow we get 9 frames from this
                [1,0,2]
            ),
            tf.repeat(tf.reshape(y, (1,RANGE)), repeats=math.floor(chunk_size/ORIGINAL_PAPER_SAMPLES_PER_CHUNK), axis=0) # Repeat our one hot tensor 9 times
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    test_ds = test_ds.map(
        lambda x, y:
        (
            tf.transpose(
                tf.signal.frame(x, ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ORIGINAL_PAPER_SAMPLES_PER_CHUNK), # Somehow we get 9 frames from this
                [1,0,2]
            ),
            tf.repeat(tf.reshape(y, (1,RANGE)), repeats=math.floor(chunk_size/ORIGINAL_PAPER_SAMPLES_PER_CHUNK), axis=0) # Repeat our one hot tensor 9 times
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    train_ds = train_ds.unbatch()
    val_ds = val_ds.unbatch()
    test_ds = test_ds.unbatch()

    train_ds = train_ds.shuffle(BATCH*NUM_REPEATS*4)
    val_ds   = val_ds.shuffle(BATCH*NUM_REPEATS*4)
    test_ds  = test_ds.shuffle(BATCH*NUM_REPEATS*4)

    # for e in test_ds:
    #     print(e[1])

    # sys.exit(1)

    train_ds = train_ds.batch(REBATCH)
    val_ds   = val_ds.batch(REBATCH)
    test_ds  = test_ds.batch(REBATCH)

    train_ds = train_ds.prefetch(100)
    val_ds   = val_ds.prefetch(100)
    test_ds  = test_ds.prefetch(100)

    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    start_time = time.time()

    # Hyper Parameters
    RANGE   = len(ALL_SERIAL_NUMBERS)
    EPOCHS  = 75
    DROPOUT = 0.5 # [0,1], the chance to drop an input
    set_seeds(1337)


    # train_ds, val_ds, test_ds = get_all_shuffled()
    # train_ds, val_ds, test_ds = get_all_shuffled_windowed()
    train_ds, val_ds, test_ds = get_shuffled_and_windowed_from_pregen_ds()
    # train_ds, val_ds, test_ds = get_less_limited_oracle()
    # train_ds, val_ds, test_ds = get_windowed_less_limited_oracle()
    # train_ds, val_ds, test_ds = get_windowed_foxtrot_shuffled()

    # train_ds = train_ds.unbatch().batch(1).take(1).cache().prefetch(100)
    # val_ds   = val_ds.unbatch().batch(1).take(1).cache().prefetch(100)
    # test_ds  = test_ds.unbatch().batch(1).take(1).cache().prefetch(100)

    # train_ds = train_ds.take(10).cache().prefetch(100)
    # val_ds   = val_ds.take(2).cache().prefetch(100)
    # test_ds  = test_ds.take(2).cache().prefetch(100)

    # train_ds = train_ds.cache().prefetch(100)
    # val_ds   = val_ds.cache().prefetch(100)
    # test_ds  = test_ds.cache().prefetch(100)

    # for e in train_ds.unbatch():
    #     print( e[1].numpy() )

    # sys.exit(1)

    inputs  = keras.Input(shape=(2,ORIGINAL_PAPER_SAMPLES_PER_CHUNK))

    x = keras.layers.Convolution1D(
        filters=50,
        kernel_size=7,
        strides=1,
        activation="relu",
        kernel_initializer='glorot_uniform',
        data_format="channels_first",
        name="classifier_3"
    )(inputs)

    x = keras.layers.Convolution1D(
        filters=50,
        kernel_size=7,
        strides=2,
        activation="relu",
        kernel_initializer='glorot_uniform',
        data_format="channels_first",
        name="classifier_4"
    )(x)

    x = keras.layers.Dropout(DROPOUT)(x)

    x = keras.layers.Flatten(name="classifier_5")(x)

    x = keras.layers.Dense(
            units=256,
            activation='relu',
            kernel_initializer='he_normal',
            name="classifier_6"
    )(x)

    x = keras.layers.Dropout(DROPOUT)(x)

    x = keras.layers.Dense(
        units=80,
        activation='relu',
        kernel_initializer='he_normal',
        name="classifier_7"
    )(x)

    x = keras.layers.Dropout(DROPOUT)(x)

    outputs = keras.layers.Dense(RANGE, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="steves_model")
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                # loss=tf.keras.losses.MeanSquaredError(), # This may do better with categorical_crossentropy
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()], # Categorical is needed for one hot encoded data
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="model_checkpoint",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_loss", # We could theoretically monitor the val loss as well (eh)
            verbose=0,
        ),
        # keras.callbacks.TensorBoard(log_dir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    ]

    history = model.fit(
        x=train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        # callbacks=callbacks,
    )


    print("Now we evaluate on the test data")
    results = model.evaluate(
        test_ds,
        verbose=1,
    )
    print("test loss:", results[0], ", test acc:", results[1])

    with open("RESULTS", "w") as f:
        f.write("Experiment name: {}\n".format(EXPERIMENT_NAME))
        f.write("test loss:{}, test acc:{}\n".format(results[0], results[1]))

    print("Now we evaluate on the val data")
    results = model.evaluate(
        val_ds,
        verbose=1,
    )
    print("val loss:", results[0], ", val acc:", results[1])

    with open("RESULTS", "a") as f:
        f.write("val loss:{}, val acc:{}\n".format(results[0], results[1]))

    test_y_hat = []
    test_y     = []

    print("Calculate the confusion matrix")
    total_confusion = None
    f = None
    for e in test_ds:
        confusion = tf.math.confusion_matrix(
            np.argmax(e[1].numpy(), axis=1),
            np.argmax(model.predict(e[0]), axis=1),
            num_classes=RANGE
        )

        if total_confusion == None:
            total_confusion = confusion
        else:
            total_confusion = total_confusion + confusion

    #plot_confusion_matrix(confusion)
    save_confusion_matrix(confusion)


    # Loss curve
    #plot_loss_curve(history)
    save_loss_curve(history)

    end_time = time.time()

    with open("RESULTS", "a") as f:
        f.write("total time seconds: {}\n".format(end_time-start_time))