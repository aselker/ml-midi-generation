#!/usr/bin/env python3
import pretty_midi
import numpy as np
import sys
import os


def midi_files_in_dir(dir_name):
    files = []
    for _, _, fs in os.walk(dir_name):
        files += [os.path.join(dir_name, f) for f in fs]
    return files


def generate_input_and_target(dict_keys_time, seq_len=50):
    """ Generate input and the target of our deep learning for one music.
    
    Parameters
    ==========
    dict_keys_time : dict
      Dictionary of timestep and notes
    seq_len : int
      The length of the sequence
      
    Returns
    =======
    Tuple of list of input and list of target of neural network.
    
       
    """
    assert dict_keys_time
    # Get the start time and end time
    start_time, end_time = (
        list(dict_keys_time.keys())[0],
        list(dict_keys_time.keys())[-1],
    )
    list_training, list_target = [], []
    for index_enum, time in enumerate(range(start_time, end_time)):
        list_append_training, list_append_target = [], []
        start_iterate = 0
        flag_target_append = False  # flag to append the test list
        if index_enum < seq_len:
            start_iterate = seq_len - index_enum - 1
            for i in range(start_iterate):  # add 'e' to the seq list.
                list_append_training.append([])
                flag_target_append = True

        for i in range(start_iterate, seq_len):
            index_enum = time - (seq_len - i - 1)
            if index_enum in dict_keys_time:
                list_append_training.append(list(dict_keys_time[index_enum]))
            else:
                list_append_training.append([])

        # add time + 1 to the list_append_target
        if time + 1 in dict_keys_time:
            list_append_target.append(list(dict_keys_time[time + 1]))
        else:
            list_append_target.append([])
        list_training.append(list_append_training)
        list_target.append(list_append_target)
    return list_training, list_target


def generate_dict_time_notes(
    list_all_midi, batch_song=16, start_index=0, fs=30, use_tqdm=True
):
    """ Generate map (dictionary) of music ( in index ) to piano_roll (in np.array)
    Parameters
    ==========
    list_all_midi : list
        List of midi files
    batch_music : int
      A number of music in one batch
    start_index : int
      The start index to be batched in list_all_midi
    fs : int
      Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    use_tqdm : bool
      Whether to use tqdm or not in the function
    Returns
    =======
    dictionary of music to piano_roll (in np.array)
    """
    assert len(list_all_midi) >= batch_song
    dict_time_notes = {}
    process_tqdm_midi = (
        tqdm_notebook(
            range(start_index, min(start_index + batch_song, len(list_all_midi)))
        )
        if use_tqdm
        else range(start_index, min(start_index + batch_song, len(list_all_midi)))
    )
    for i in process_tqdm_midi:
        midi_file_name = list_all_midi[i]
        if use_tqdm:
            process_tqdm_midi.set_description("Processing {}".format(midi_file_name))
        try:  # Handle exception on malformat MIDI files
            midi_pretty_format = pretty_midi.PrettyMIDI(midi_file_name)
            piano_midi = midi_pretty_format.instruments[0]  # Get the piano channels
            piano_roll = piano_midi.get_piano_roll(fs=fs)
            if len(piano_roll) == 0:
                print("File {} is has no notes".format(midi_file_name))
            else:
                dict_time_notes[i] = piano_roll
        except Exception as e:
            print(e)
            print("broken file : {}".format(midi_file_name))
            pass
    return dict_time_notes


def process_notes_in_song(dict_time_notes, seq_len=50):
    """
    Iterate the dict of piano rolls into dictionary of timesteps and note played
    
    Parameters
    ==========
    dict_time_notes : dict
      dict contains index of music ( in index ) to piano_roll (in np.array)
    seq_len : int
      Length of the sequence
      
    Returns
    =======
    Dict of timesteps and note played
    """
    list_of_dict_keys_time = []

    for key, sample in dict_time_notes.items():
        times = np.unique(np.where(sample > 0)[1])
        index = np.where(sample > 0)
        dict_keys_time = {}

        for time in times:
            index_where = np.where(index[1] == time)
            notes = index[0][index_where]
            dict_keys_time[time] = notes
        if len(times) == 0:
            print("Song is empty")
        else:
            list_of_dict_keys_time.append(dict_keys_time)
    return list_of_dict_keys_time


def generate_batch_song(
    list_all_midi, batch_music=16, start_index=0, fs=30, seq_len=50, use_tqdm=False
):
    """
    Generate Batch music that will be used to be input and output of the neural network
    
    Parameters
    ==========
    list_all_midi : list
      List of midi files
    batch_music : int
      A number of music in one batch
    start_index : int
      The start index to be batched in list_all_midi
    fs : int
      Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    seq_len : int
      The sequence length of the music to be input of neural network
    use_tqdm : bool
      Whether to use tqdm or not in the function
    
    Returns
    =======
    Tuple of input and target for neural network
    
    """

    assert len(list_all_midi) >= batch_music
    print("Generating time notes...")
    dict_time_notes = generate_dict_time_notes(
        list_all_midi, batch_music, start_index, fs, use_tqdm=use_tqdm
    )

    print("Processing notes...")
    list_musics = process_notes_in_song(dict_time_notes, seq_len)
    collected_list_input, collected_list_target = [], []

    print("Generating input and target sets...")
    for music in list_musics:
        list_training, list_target = generate_input_and_target(music, seq_len)
        collected_list_input += list_training
        collected_list_target += list_target
    return collected_list_input, collected_list_target


if __name__ == "__main__":
    files = midi_files_in_dir(sys.argv[1])
    print(generate_batch_song(files))
