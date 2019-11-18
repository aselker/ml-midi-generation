#!/usr/bin/env python3

import pretty_midi
import sys
import numpy as np
import os, os.path
import pickle


def midi_to_num(midifiles):
    dict_notes = {}
    for i in midifiles:
        try:
            midi = pretty_midi.PrettyMIDI(i)
        except IOError:
            continue
        piano_roll = midi.instruments[0].get_piano_roll()
        piano_roll = np.transpose(piano_roll)
        dict_notes[i] = piano_roll
    return dict_notes


def all_midis(rootdir):
    midiFiles = []
    for subdir, ___, filename in os.walk(rootdir):
        for files in filename:
            if files.endswith(".mid"):
                midiFiles.append(os.path.join(subdir, files))
    return midiFiles


def get_random_batch(songs, n):
    s = list(songs.values())
    songs_to_use = np.random.choice(len(s), size=n, replace=False)
    return [s[song] for song in songs_to_use]


if __name__ == "__main__":
    print("Reading file...")
    songs = midi_to_num([sys.argv[1]])
    if songs == {}:
        print("File is empty.")
    else:
        print("Saving...")
        song = list(songs.values())[0]
        pickle.dump(song, open(sys.argv[2], "wb"))
