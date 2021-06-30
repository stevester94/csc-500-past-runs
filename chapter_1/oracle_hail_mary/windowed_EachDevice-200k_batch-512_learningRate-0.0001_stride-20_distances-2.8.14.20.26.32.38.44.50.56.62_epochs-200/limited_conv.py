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

EXPERIMENT_NAME = "windowed_EachDevice-200k_batch-512_learningRate-0.0001_stride-20_distances-2.8.14.20.26.32.38.44.50.56.62_epochs-200"
LEARNING_RATE = 0.0001
ORIGINAL_BATCH_SIZE=100
DESIRED_BATCH_SIZE=512
EPOCHS  = 1000

# Setting the seed is vital for reproducibility
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def get_shuffled_and_windowed_from_pregen_ds():
    from steves_utils.ORACLE.windowed_shuffled_dataset_accessor import Windowed_Shuffled_Dataset_Factory
    from steves_utils import utils

    path = os.path.join(utils.get_datasets_base_path(), "automated_windower", "windowed_EachDevice-200k_batch-100_stride-20_distances-2.8.14.20.26.32.38.44.50.56.62")
    print(path)
    datasets = Windowed_Shuffled_Dataset_Factory(path)

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

    train_ds = train_ds.unbatch()
    val_ds = val_ds.unbatch()
    test_ds = test_ds.unbatch()

    train_ds = train_ds.shuffle(100 * ORIGINAL_BATCH_SIZE, reshuffle_each_iteration=True)
    
    train_ds = train_ds.batch(DESIRED_BATCH_SIZE)
    val_ds  = val_ds.batch(DESIRED_BATCH_SIZE)
    test_ds = test_ds.batch(DESIRED_BATCH_SIZE)

    train_ds = train_ds.prefetch(100)
    val_ds   = val_ds.prefetch(100)
    test_ds  = test_ds.prefetch(100)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    start_time = time.time()

    # Hyper Parameters
    RANGE   = len(ALL_SERIAL_NUMBERS)
    DROPOUT = 0.5 # [0,1], the chance to drop an input
    set_seeds(1337)

    train_ds, val_ds, test_ds = get_shuffled_and_windowed_from_pregen_ds()

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

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                # loss=tf.keras.losses.MeanSquaredError(), # This may do better with categorical_crossentropy
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()], # Categorical is needed for one hot encoded data
    )

    # callbacks = [
    #     keras.callbacks.ModelCheckpoint(
    #         filepath="model_checkpoint",
    #         save_best_only=True,  # Only save a model if `val_loss` has improved.
    #         monitor="val_loss", # We could theoretically monitor the val loss as well (eh)
    #         verbose=0,
    #     ),
    #     # keras.callbacks.TensorBoard(log_dir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    # ]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    ]

    history = model.fit(
        x=train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
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