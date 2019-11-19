#!/usr/bin/env python3

import sys
import numpy as np
import torch as t
from torch import nn

from rnn_model import RnnModel

print("Loading saved model...")
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
saved_data = t.load(sys.argv[1], map_location=device)
params = saved_data["params"]
data_width, _, state_size, n_layers = tuple(params)

model = RnnModel(*params)
model.load_state_dict(saved_data["state_dict"])


song = np.zeros((1, data_width))
state = t.zeros(n_layers, 1, state_size)

for _ in range(100):
    input_ = t.tensor([[song[-1]]], dtype=t.float32)
    output, state = model(input_, state)
    next_notes = output > t.rand(data_width)
    print(next_notes)
    song = np.append(song, next_notes, axis=0)

print(song.shape)
# TODO: Convert matrix to midi file
