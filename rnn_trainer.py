#!/usr/bin/env python3

# Heavily based on / slightly copied from:
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

import sys
import numpy as np
import torch as t
from torch import nn
import torch.utils.data
import pickle
import random

import midi_to_num
from rnn_model import RnnModel

assert sys.argv[1]
assert sys.argv[2]

dtype = np.float32
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")


def prep_data(seq, data_width):
    # Even out the lengths of input_seq and target_seq
    # See https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e

    seq_lens = [len(s) for s in seq]
    pad_token = -1
    current_batch_size = len(seq)
    padded = np.ones((current_batch_size, max(seq_lens), data_width)) * pad_token

    for i, l in enumerate(seq_lens):
        padded[i, :l] = seq[i][:l]

    # Create input and target datasets
    input_padded = np.array([x[:-1] for x in padded])
    target_padded = np.array([x[1:] for x in padded])

    input_padded = t.Tensor(input_padded)
    target_padded = t.Tensor(target_padded)
    input_padded = input_padded.to(device)
    target_padded = target_padded.to(device)

    return input_padded, target_padded


# Define parameters
data_width = 128
state_size = 200
n_layers = 4
n_epochs = 5
n_batches = 50
test_portion = 0.04
lr = 0.001

all_files = all_pklmidis(sys.argv[1])
all_files = unpkl(all_files)
random.shuffle(all_files)
data_count = len(all_files)
print("Using {} files".format(data_count))

# Split off some data for testing
test_data_count = int(data_count * test_portion)
train_files = all_files[:-test_data_count]
test_files = all_files[-test_data_count:]

model = RnnModel(data_width, data_width, state_size, n_layers)
model = model.to(device)


loss_criterion = nn.SmoothL1Loss()
optimizer = t.optim.Adam(model.parameters(), lr=lr)

# Make batch sizes
remaining_data_len = len(train_files)
batch_sizes = [-1 for _ in range(n_batches)]
for i, b in enumerate(range(n_batches, 0, -1)):
    batch_sizes[i] = round(remaining_data_len / b)
    remaining_data_len -= batch_sizes[i]

for epoch in range(n_epochs):
    shuffled_files = all_files
    random.shuffle(shuffled_files)
    batches = []

    for size in batch_sizes:
        batches.append(shuffled_files[-size:])
        shuffled_files = shuffled_files[:-size]

    for i, file_names in enumerate(batches):
        seq = list(file_names)

        input_padded, target_padded = prep_data(seq, data_width)
        # Run the model
        print("Running model...")
        optimizer.zero_grad()
        # Initialize the state to zeros
        state = t.zeros(n_layers, len(seq), state_size)
        state = state.to(device)
        output, _ = model(input_padded, state)

        loss = loss_criterion(output.view(-1), target_padded.view(-1))
        loss.backward()
        optimizer.step()
        print("Finished batch {}.".format(i))

    test_data = list(test_files)
    test_input, test_target = prep_data(test_data, data_width)
    test_state = t.zeros(n_layers, len(test_data), state_size)
    test_state = test_state.to(device)
    test_output, _ = model(test_input, test_state)
    test_loss = loss_criterion(test_output.view(-1), test_target.view(-1))
    del (test_data, test_input, test_target)  # Save a little memory

    if epoch % 1 == 0:
        print(
            "Epoch {}; training loss: {}  Testing loss: {}".format(
                epoch, loss.item(), test_loss.item()
            )
        )

params = [data_width, data_width, state_size, n_layers]
t.save({"state_dict": model.state_dict(), "params": params}, sys.argv[2])
