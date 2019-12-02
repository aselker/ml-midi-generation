#!/usr/bin/env python3

import sys
import numpy as np
import torch as t
from torch import nn

from rnn_model import RnnModel
from piano_roll_to_pretty_midi import piano_roll_to_pretty_midi
from numpy_to_midiutil import make_midi

assert sys.argv[1]
assert sys.argv[2]

print("Loading saved model...")
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
saved_data = t.load(sys.argv[1], map_location=device)
params = saved_data["params"]
data_width, _, state_size, n_layers = tuple(params)

model = RnnModel(*params)
model.load_state_dict(saved_data["state_dict"])


song = np.zeros((1, data_width))
state = None

for i in range(1000):
    input_ = t.tensor([[song[-1]]], dtype=t.float32)
    output, state = model(input_, state)

    output_fuzzy = output.detach().numpy()
    output_fuzzy += np.random.rand(data_width) * (2 * (1 + np.sin(i * 2 * np.pi / 16)))
    argsorted = output_fuzzy.argsort()
    notes_to_play = argsorted[0][-4:]
    next_notes = np.zeros(data_width)
    for note in notes_to_play:
        next_notes[note] = 1
    next_notes = [next_notes]
    print(next_notes)

    song = np.append(song, next_notes, axis=0)

make_midi(song, sys.argv[2])
