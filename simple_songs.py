#!/usr/bin/env python3
import numpy as np


def to_one_hot(xs):
    a_s = []
    for x in xs:
        a = [0 for _ in range(8)]
        if a:
            a[x - 1] = 1
        a_s.append(a)
    return np.array(a_s)


ode_to_joy = to_one_hot(
    [
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
)
