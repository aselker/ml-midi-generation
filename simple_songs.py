#!/usr/bin/env python3

import sys
import numpy as np
import pickle


def to_one_hot(xs):
    a_s = []
    for x in xs:
        a = [0 for _ in range(128)]
        if a:
            a[x - 1] = 1
        a_s.append(a)
    return np.array(a_s)


notes = [
    3,
    3,
    4,
    5,
    5,
    4,
    3,
    2,
    1,
    1,
    2,
    3,
    3,
    2,
    2,
    0,
    3,
    3,
    4,
    5,
    5,
    4,
    3,
    2,
    1,
    1,
    2,
    3,
    2,
    1,
    1,
    0,
    2,
    2,
    3,
    1,
    2,
    4,
    3,
    1,
    2,
    4,
    3,
    2,
    1,
    2,
    5,
    0,
    3,
    3,
    4,
    5,
    5,
    4,
    3,
    2,
    1,
    1,
    2,
    3,
    2,
    1,
    1,
]
notes = np.array(notes) + 64

ode_to_joy = to_one_hot(list(notes) * 20)


pickle.dump(ode_to_joy, open(sys.argv[1], "wb"))
