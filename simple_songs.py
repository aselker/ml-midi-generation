#!/usr/bin/env python3

import sys
import numpy as np
import pickle


def to_one_hot(xs):
    a_s = []
    for x in xs:
        a = [0 for _ in range(128)]
        if x:
            a[x + 63] = 1
        a_s.append(a)
    return np.array(a_s)


notes = [
    5,
    5,
    6,
    8,
    8,
    6,
    5,
    3,
    1,
    1,
    3,
    5,
    5,
    3,
    3,
    0,
    5,
    5,
    6,
    8,
    8,
    6,
    5,
    3,
    1,
    1,
    3,
    5,
    3,
    1,
    1,
    0,
    3,
    3,
    5,
    1,
    3,
    6,
    5,
    1,
    3,
    6,
    5,
    3,
    1,
    3,
    8,
    0,
    5,
    5,
    6,
    8,
    8,
    6,
    5,
    3,
    1,
    1,
    3,
    5,
    3,
    1,
    1,
]

ode_to_joy = to_one_hot(list(notes) * 20)

if __name__ == "__main__":
    pickle.dump(ode_to_joy, open(sys.argv[1], "wb"))
