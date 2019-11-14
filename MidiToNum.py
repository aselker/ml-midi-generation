import pretty_midi
import sys
import numpy as np
import os, os.path


def MidiToNum(midifiles):
    dict_notes = {}
    for i in midifiles:
        try:
            midi = pretty_midi.PrettyMIDI(i)
        except IOError:
            continue
        piano_roll = midi.instruments[0].get_piano_roll()
        dict_notes[i] = piano_roll
    return dict_notes


def AllMidis(rootdir):
    midiFiles = []
    for subdir, ___, filename in os.walk(rootdir):
        for files in filename:
            if files.endswith(".mid"):
                midiFiles.append(os.path.join(subdir, files))
    return midiFiles


if __name__ == "__main__":
    files = AllMidis(sys.argv[1])
    notes = MidiToNum(files)
    print(notes)
