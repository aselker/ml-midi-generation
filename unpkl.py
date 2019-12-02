#!/usr/bin/env python3

import os, os.path
import pickle
import multiprocessing as mp


def all_pklmidis(rootdir):
    pickle_files = []
    for subdir, ___, filename in os.walk(rootdir):
        for files in filename:
            if files.endswith(".mid") or files.endswith(".pkl"):
                pickle_files.append(os.path.join(subdir, files))
    return pickle_files


def f(pair):
    i, l = pair
    print("Unpickling file {}.".format(i))
    return pickle.load(open(l, "rb"))


def unpkl(midis):

    # files = []
    # for i, l in enumerate(midis):
    # print("Unpickling file {}.".format(i))
    # files.append(pickle.load(open(l, "rb")))

    pool = mp.Pool(processes=mp.cpu_count())
    files = pool.map(f, enumerate(midis))
    return files
