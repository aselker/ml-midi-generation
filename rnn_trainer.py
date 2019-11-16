#!/usr/bin/env python3

# Heavily based on / slightly copied from:
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

import sys
import numpy as np
import torch as t
from torch import nn
import torch.utils.data
import pickle

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

    seq = np.asarray(seq)
    for i, l in enumerate(seq_lens):
        padded[i, :l] = seq[i, :l]

    # Create input and target datasets
    input_padded = [x[:-1] for x in padded]
    target_padded = [x[1:] for x in padded]

    input_padded = t.Tensor(input_padded)
    target_padded = t.Tensor(target_padded)
    input_padded = input_padded.to(device)
    target_padded = target_padded.to(device)

    return input_padded, target_padded


# Define hyperparameters
state_size = 200
n_layers = 4
n_epochs = 30
lr = 0.01

all_data = pickle.load(open(sys.argv[1], "rb"))
np.random.shuffle(all_data)

data_count = len(all_data)
data_width = len(all_data[0][0])

# Split off some data for testing
test_data_count = int(data_count / 5)
train_data = all_data[:-test_data_count]
test_data = all_data[-test_data_count:]

model = RnnModel(data_width, data_width, state_size, n_layers)
model = model.to(device)


loss_criterion = nn.SmoothL1Loss()
optimizer = t.optim.Adam(model.parameters(), lr=lr)

remaining_data_len = len(train_data)
n_batches = len(train_data)
batch_sizes = [-1 for _ in range(n_batches)]
for i, b in enumerate(range(n_batches, 0, -1)):
    batch_sizes[i] = round(remaining_data_len / b)
    remaining_data_len -= batch_sizes[i]

for epoch in range(n_epochs):

    # Split training data into mini-batches
    batches = torch.utils.data.random_split(train_data, batch_sizes)
    for seq in batches:
        input_padded, target_padded = prep_data(seq, data_width)
        # Run the model
        optimizer.zero_grad()
        # Initialize the state to zeros
        state = t.zeros(n_layers, len(seq), state_size)
        state = state.to(device)
        output, _ = model(input_padded, state)

        loss = loss_criterion(output.view(-1), target_padded.view(-1))
        loss.backward()
        optimizer.step()

    # TODO: Test loss on test dataset

    if epoch % 1 == 0:
        print(
            "Epoch {}; training loss: {}  Testing loss: {}".format(
                epoch, loss.item(), test_loss.item()
            )
        )

params = [data_width, data_width, 200, 4]

t.save({"state_dict": model.state_dict(), "params": params}, sys.argv[2])
