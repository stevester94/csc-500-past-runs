#! /usr/bin/env python3

import random
import time
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
from test import test
import json

from tf_dataset_getter  import get_shuffled_and_windowed_from_pregen_ds


start_time_secs = time.time()

torch.set_default_dtype(torch.float64)


import matplotlib.pyplot as plt
def _do_loss_curve(history):
    figure, axis = plt.subplots(2, 1)

    figure.set_size_inches(12, 12)
    figure.suptitle("Loss During Training")
    plt.subplots_adjust(hspace=0.4)
    plt.rcParams['figure.dpi'] = 600
    
    axis[0].set_title("Label Loss")
    axis[0].plot(history["indices"], history['source_val_label_loss'], label='Source Validation Label Loss')
    axis[0].plot(history["indices"], history['source_train_label_loss'], label='Source Train Label Loss')
    axis[0].plot(history["indices"], history['target_val_label_loss'], label='Target Validation Label Loss')
    axis[0].legend()
    axis[0].grid()
    axis[0].set(xlabel='Epoch', ylabel="CrossEntropy Loss")
    axis[0].locator_params(axis="x", integer=True, tight=True)
    
    # axis[0].xlabel('Epoch')

    axis[1].set_title("Domain Loss")
    axis[1].plot(history["indices"], history['target_val_domain_loss'], label='Source Validation Domain Loss')
    axis[1].plot(history["indices"], history['source_train_domain_loss'], label='Source Train Domain Loss')
    axis[1].plot(history["indices"], history['source_val_domain_loss'], label='Target Validation Domain Loss')
    axis[1].legend()
    axis[1].grid()
    axis[1].set(xlabel='Epoch', ylabel="L1 Loss")
    axis[1].locator_params(axis="x", integer=True, tight=True)


def plot_loss_curve(history):
    _do_loss_curve(history)
    plt.show()

def save_loss_curve(history, path="./loss_curve.png"):
    _do_loss_curve(history)
    plt.savefig(path)

BATCH_LOGGING_DECIMATION_FACTOR = 20
BEST_MODEL_PATH = "./best_model.pth"
cuda = True
cudnn.benchmark = True

lr = 0.0001
n_epoch = 2000
batch_size = 128
source_distance = "2.8.14.20.26"
target_distance = 32
alpha = 0.001
num_additional_extractor_fc_layers=1
experiment_name = "Fill Me ;)"
patience = 10

if __name__ == "__main__" and len(sys.argv) == 1:
    j = json.loads(sys.stdin.read())

    lr = j["lr"]
    n_epoch = j["n_epoch"]
    batch_size = j["batch_size"]
    source_distance = j["source_distance"]
    target_distance = j["target_distance"]
    alpha = j["alpha"]
    num_additional_extractor_fc_layers = j["num_additional_extractor_fc_layers"]
    experiment_name = j["experiment_name"]
    patience = j["patience"]

    print("experiment_name:", experiment_name)
    print("lr:", lr)
    print("n_epoch:", n_epoch)
    print("batch_size:", batch_size)
    print("source_distance:", source_distance)
    print("target_distance:", target_distance)
    print("alpha:", alpha)
    print("num_additional_extractor_fc_layers:", num_additional_extractor_fc_layers)
    print("patience:", patience)


manual_seed = 1337
random.seed(manual_seed)
torch.manual_seed(manual_seed)

from steves_utils import utils

# batch_size = 1
ORIGINAL_BATCH_SIZE = 100

source_ds_path = "{datasets_base_path}/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-{distance}".format(
    datasets_base_path=utils.get_datasets_base_path(), distance=source_distance
)

target_ds_path = "{datasets_base_path}/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-{distance}".format(
    datasets_base_path=utils.get_datasets_base_path(), distance=target_distance
)



train_ds_source, val_ds_source, test_ds_source = get_shuffled_and_windowed_from_pregen_ds(source_ds_path, ORIGINAL_BATCH_SIZE, batch_size)
train_ds_target, val_ds_target, test_ds_target = get_shuffled_and_windowed_from_pregen_ds(target_ds_path, ORIGINAL_BATCH_SIZE, batch_size)



print("Unfortunately have to calculate the length of the source dataset by iterating over it. Standby...")
num_batches_in_train_ds_source = 0
for i in train_ds_source:
    num_batches_in_train_ds_source += 1
print("Done. Source Train DS Length:", num_batches_in_train_ds_source)

print("Unfortunately have to calculate the length of the source dataset by iterating over it. Standby...")
num_batches_in_train_ds_target = 0
for i in train_ds_target:
    num_batches_in_train_ds_target += 1
print("Done. Target Train DS Length:", num_batches_in_train_ds_target)

# print("We are hardcoding DS length!")
# num_batches_in_train_ds_source = 25000
# num_batches_in_train_ds_target = 25000
# num_batches_in_train_ds_source = 250
# num_batches_in_train_ds_target = 250

my_net = CNNModel(num_additional_extractor_fc_layers)

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.CrossEntropyLoss()
# loss_domain = torch.nn.CrossEntropyLoss()
loss_domain = torch.nn.L1Loss()
# loss_domain = torch.nn.MSELoss()


if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training
best_accu_t = 0.0
last_time = time.time()


history = {}
history["indices"] = []
history["source_val_label_loss"] = []
history["source_val_domain_loss"] = []
history["target_val_label_loss"] = []
history["target_val_domain_loss"] = []
history["source_train_label_loss"] = []
history["source_train_domain_loss"] = []
history["source_val_label_accuracy"] = []
history["target_val_label_accuracy"] = []

best_epoch_index_and_combined_val_label_loss = [0, float("inf")]
for epoch in range(1,n_epoch+1):

    data_source_iter = train_ds_source.as_numpy_iterator()
    # data_target_iter = train_ds_target.as_numpy_iterator()

    err_s_label_epoch = 0
    err_s_domain_epoch = 0

    for i in range(num_batches_in_train_ds_source):

        if alpha is None:
            p = float(i + epoch * num_batches_in_train_ds_source) / n_epoch / num_batches_in_train_ds_source
            gamma = 10
            alpha = 2. / (1. + np.exp(-gamma * p)) - 1

        # alpha = 0
        # print(p)

        # print("Alpha", alpha)

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label, s_domain = data_source

        s_img = torch.from_numpy(s_img)
        s_label = torch.from_numpy(s_label).long()
        s_domain = torch.from_numpy(s_domain).long()

        my_net.zero_grad()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            s_domain = s_domain.cuda()


        class_output, domain_output = my_net(input_data=s_img, t=s_domain, alpha=alpha)
        domain_output = torch.flatten(domain_output)


        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(domain_output, s_domain)

        err_s_label_epoch += err_s_label.cpu().item()
        err_s_domain_epoch += err_s_domain.cpu().item()

        err = err_s_domain + err_s_label


        err.backward()
        optimizer.step()

        if i % BATCH_LOGGING_DECIMATION_FACTOR == 0:
            cur_time = time.time()
            batches_per_second = BATCH_LOGGING_DECIMATION_FACTOR / (cur_time - last_time)
            last_time = cur_time
            sys.stdout.write(
                (
                    "epoch: {epoch}, [iter: {batch} / all {total_batches}], "
                    "batches_per_second: {batches_per_second:.4f}, "
                    "err_s_label: {err_s_label:.4f}, "
                    "err_s_domain: {err_s_domain:.4f}, "
                    "alpha: {alpha:.4f}\n"
                ).format(
                        batches_per_second=batches_per_second,
                        epoch=epoch,
                        batch=i,
                        total_batches=num_batches_in_train_ds_source,
                        err_s_label=err_s_label.cpu().item(),
                        err_s_domain=err_s_domain.cpu().item(),
                        alpha=alpha
                    )
            )

            sys.stdout.flush()

    source_val_label_accuracy, source_val_label_loss, source_val_domain_loss = \
        test(my_net, loss_class, loss_domain, val_ds_source.as_numpy_iterator())
    
    target_val_label_accuracy, target_val_label_loss, target_val_domain_loss = \
        test(my_net, loss_class, loss_domain, val_ds_target.as_numpy_iterator())

    history["indices"].append(epoch)
    history["source_val_label_loss"].append(source_val_label_loss)
    history["source_val_domain_loss"].append(source_val_domain_loss)
    history["target_val_label_loss"].append(target_val_label_loss)
    history["target_val_domain_loss"].append(target_val_domain_loss)
    history["source_train_label_loss"].append(err_s_label_epoch / i)
    history["source_train_domain_loss"].append(err_s_domain_epoch / i)
    history["source_val_label_accuracy"].append(source_val_label_accuracy)
    history["target_val_label_accuracy"].append(target_val_label_accuracy)

    sys.stdout.write(
        (
            "=============================================================\n"
            "epoch: {epoch}, "
            "acc_src_val_label: {source_val_label_accuracy:.4f}, "
            "err_src_val_label: {source_val_label_loss:.4f}, "
            "err_src_val_domain: {source_val_domain_loss:.4f}, "
            "acc_trgt_val_label: {target_val_label_accuracy:.4f}, "
            "err_trgt_val_label: {target_val_label_loss:.4f}, "
            "err_trgt_val_domain: {target_val_domain_loss:.4f}"
            "\n"
            "=============================================================\n"
        ).format(
                epoch=epoch,
                source_val_label_accuracy=source_val_label_accuracy,
                source_val_label_loss=source_val_label_loss,
                source_val_domain_loss=source_val_domain_loss,
                target_val_label_accuracy=target_val_label_accuracy,
                target_val_label_loss=target_val_label_loss,
                target_val_domain_loss=target_val_domain_loss,
            )
    )

    sys.stdout.flush()

    combined_val_label_loss = source_val_label_loss + target_val_label_loss
    if best_epoch_index_and_combined_val_label_loss[1] > combined_val_label_loss:
        print("New best")
        best_epoch_index_and_combined_val_label_loss[0] = epoch
        best_epoch_index_and_combined_val_label_loss[1] = combined_val_label_loss
        torch.save(my_net, BEST_MODEL_PATH)
    
    elif epoch - best_epoch_index_and_combined_val_label_loss[0] > patience:
        print("Patience ({}) exhausted".format(patience))
        break


print("Loading best model from epoch {} with combined loss of {}".format(*best_epoch_index_and_combined_val_label_loss))
my_net = torch.load(BEST_MODEL_PATH)

save_loss_curve(history)
    # accu_t = test(test_ds_target)
    # print('Accuracy of the %s dataset: %f\n' % ('Target', accu_t))
    # if accu_t > best_accu_t:
    #     best_accu_s = accu_s
    #     best_accu_t = accu_t

source_test_label_accuracy, source_test_label_loss, source_test_domain_loss = \
    test(my_net, loss_class, loss_domain, val_ds_source.as_numpy_iterator())

target_test_label_accuracy, target_test_label_loss, target_test_domain_loss = \
    test(my_net, loss_class, loss_domain, val_ds_target.as_numpy_iterator())

stop_time_secs = time.time()
total_time_secs = stop_time_secs - start_time_secs

with open("results.txt", "w") as f:
    out = ""
    out += "Experiment name: {}\n".format(
        experiment_name
    )

    out += "Source Test Label Acc: {test_label_acc}, Source Test Label Loss: {test_label_loss}, Source Test Domain Loss: {test_domain_loss}\n".format(
        test_label_acc=source_test_label_accuracy,
        test_label_loss=source_test_label_loss,
        test_domain_loss=source_test_domain_loss,
    )

    out += "Target Test Label Acc: {test_label_acc}, Target Test Label Loss: {test_label_loss}, Target Test Domain Loss: {test_domain_loss}\n".format(
        test_label_acc=target_test_label_accuracy,
        test_label_loss=target_test_label_loss,
        test_domain_loss=target_test_domain_loss,
    )

    out += "total time seconds: {}\n".format(total_time_secs)
    print(out)
    f.write(out)


with open("results.csv", "w") as f:
    header = "NAME,SOURCE_DISTANCE,TARGET_DISTANCE,LEARNING_RATE,ALPHA,BATCH,EPOCHS,PATIENCE,"
    header += "SOURCE_TEST_LABEL_LOSS,SOURCE_TEST_LABEL_ACC,SOURCE_TEST_DOMAIN_LOSS,"
    header += "TARGET_TEST_LABEL_LOSS,TARGET_TEST_LABEL_ACC,TARGET_TEST_DOMAIN_LOSS,TOTAL_TIME_SECS\n"


    row = "{NAME},{SOURCE_DISTANCE},{TARGET_DISTANCE},{LEARNING_RATE},{ALPHA},{BATCH},{EPOCHS},{PATIENCE},"
    row += "{SOURCE_TEST_LABEL_LOSS},{SOURCE_TEST_LABEL_ACC},{SOURCE_TEST_DOMAIN_LOSS},"
    row += "{TARGET_TEST_LABEL_LOSS},{TARGET_TEST_LABEL_ACC},{TARGET_TEST_DOMAIN_LOSS},{TOTAL_TIME_SECS}\n"

    row = row.format(
        NAME=experiment_name,
        SOURCE_DISTANCE=source_distance,
        TARGET_DISTANCE=target_distance,
        LEARNING_RATE=lr,
        ALPHA=alpha,
        BATCH=batch_size,
        EPOCHS=n_epoch,
        PATIENCE=patience,
        SOURCE_TEST_LABEL_LOSS=source_test_label_loss,
        SOURCE_TEST_LABEL_ACC= source_test_label_accuracy,
        SOURCE_TEST_DOMAIN_LOSS=source_test_domain_loss,
        TARGET_TEST_LABEL_LOSS=target_test_label_loss,
        TARGET_TEST_LABEL_ACC=target_test_label_accuracy,
        TARGET_TEST_DOMAIN_LOSS=target_test_domain_loss,
        TOTAL_TIME_SECS=total_time_secs,
    )

    f.write(header)
    f.write(row)

# Save history to csv
with open("loss.csv", "w") as f:
    f.write("epoch,source_val_label_loss,source_val_domain_loss,target_val_label_loss,target_val_domain_loss,source_train_label_loss,source_train_domain_loss,source_val_label_accuracy,target_val_label_accuracy\n")
    for i in range(len(history["indices"])):
        f.write(
            ",".join(
                (
                    str(i),
                    str(history["source_val_label_loss"][i]),
                    str(history["source_val_domain_loss"][i]),
                    str(history["target_val_label_loss"][i]),
                    str(history["target_val_domain_loss"][i]),
                    str(history["source_train_label_loss"][i]),
                    str(history["source_train_domain_loss"][i]),
                    str(history["source_val_label_accuracy"][i]),
                    str(history["target_val_label_accuracy"][i]),
                )
            )
        )
        f.write("\n")