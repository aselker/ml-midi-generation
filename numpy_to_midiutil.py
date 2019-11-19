#!/usr/bin/env python3

import midiutil
import numpy as np


def make_midi(song, filename):

    my_midi = midiutil.MIDIFile(1)
    track = 0
    channel = 0
    volume = 100

    notes = get_note_properties(song)
    for pitch, start_time, duration in notes:
        my_midi.addNote(track, channel, pitch, start_time, duration, volume)

    my_midi.writeFile(open(filename, "wb"))


def get_note_properties(song):
    """
    Convert a numpy piano-roll to a list of notes, where each
    note is a tuple of (pitch, begin time, duration).
    """

    note_beginnings = np.clip(np.diff(song, axis=0, prepend=0, append=0), 0, 1)
    note_beginnings = np.argwhere(note_beginnings)

    notes = []
    for note in note_beginnings:
        start_time = note[0]
        pitch = note[1]

        this_pitch = np.transpose(song)[pitch]

        duration = 1
        while (start_time + duration < len(this_pitch)) and this_pitch[
            start_time + duration
        ]:
            duration += 1

        notes.append((pitch, start_time, duration))
    return notes


if __name__ == "__main__":
    import simple_songs

    ode = simple_songs.ode_to_joy[:32]

    make_midi(ode, "ode.midi")
