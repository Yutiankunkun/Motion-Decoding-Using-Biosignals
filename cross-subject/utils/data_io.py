# utils/data_io.py
# -*- coding: utf-8 -*-

"""Data loading and preparation utilities.

Contains functions to read .mat files, convert them into
MNE RawArray, or produce CSV segments from EEG data with events.
"""

import os
import numpy as np
import pandas as pd
from pymatreader import read_mat
import mne
import matplotlib.pyplot as plt


def load_data(file_dir: str) -> dict:
    """Load all .mat files within each subject subdirectory.

    Args:
        file_dir (str): Path to the parent directory containing subject folders.

    Returns:
        dict: A dictionary of the form.
    """
    all_data = {}

    for subject in os.listdir(file_dir):
        subject_path = os.path.join(file_dir, subject)

        if os.path.isdir(subject_path):
            all_data[subject] = {}

            for mat_file in os.listdir(subject_path):
                if mat_file.endswith('.mat'):
                    file_path = os.path.join(subject_path, mat_file)
                    
                    try:
                        mat_data = read_mat(file_path)
                        filtered_data = {key: value for key, value in mat_data.items() if not key.startswith('__')}
                        all_data[subject][mat_file] = filtered_data
                        
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    return all_data


def mne_data(raw_data: dict, sfreq: float = 500.0) -> mne.io.Raw:
    """Convert raw dict data into an MNE RawArray.

    Args:
        raw_data (dict): Dictionary containing 'data' (n_chans x n_times),
            'ch_labels', etc.
        sfreq (float): Sampling frequency.

    Returns:
        mne.io.Raw: The MNE Raw object.
    """
    ch_names = [c.replace(' ', '') for c in raw_data['ch_labels']]  # Get channel names
    ch_types = ['eeg'] * len(ch_names)  # Assume all channels are EEG

    info = mne.create_info(ch_names=ch_names, sfreq=500, ch_types=ch_types)

    raw = mne.io.RawArray(raw_data['data']*1e-6, info)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    return raw


def make_data(
    src_dir: str,
    dst_dir: str,
    subject_id: str
) -> None:
    """
    Main function to read .mat files for one subject, parse events, cut segments,
    and classify them by labels into train and validation directories.

    Args:
        src_dir (str): Source directory containing raw .mat files for subjects.
        dst_dir (str): Destination directory for storing training CSV segments.
        subject_id (str): Subject name or ID.
    """
    print(f'{subject_id} data preparation started.')
    
    os.makedirs(os.path.join(src_dir, dst_dir, 'train', subject_id), exist_ok=True)
    os.makedirs(os.path.join(src_dir, dst_dir, 'val', subject_id), exist_ok=True)

    labels = {
        '11': 'frontside_kickturn',
        '12': 'backside_kickturn',
        '13': 'pumping',
        '21': 'frontside_kickturn',
        '22': 'backside_kickturn',
        '23': 'pumping'
    }
    counts = {'frontside_kickturn': 0, 'backside_kickturn': 0, 'pumping': 0}

    for fname in os.listdir(os.path.join(src_dir, 'data', subject_id)):
        data = read_mat(os.path.join(src_dir, 'data', subject_id, fname))
        event = pd.DataFrame(data['event'])[['init_time', 'type']]

        ts = pd.DataFrame(
            np.concatenate([np.array([data['times']]), data['data']]).T, 
            columns=['Time'] + list(data['ch_labels'])
        )

        for i, d in event.iterrows():
            it = d['init_time']+0.2
            et = d['init_time']+0.7
            event_type = str(int(d['type']))
            ts_seg = ts[(ts['Time']>=it*1e3)&(ts['Time']<=et*1e3)]

            if fname!='train3.mat':
                if not os.path.exists(os.path.join(src_dir, dst_dir, 'train', subject_id, labels[event_type])):
                    os.makedirs(os.path.join(src_dir, dst_dir, 'train', subject_id, labels[event_type]), exist_ok=True)
                del ts_seg['Time']
                ts_seg.to_csv(os.path.join(src_dir, dst_dir, 'train', subject_id, labels[event_type], '{:03d}.csv'.format(counts[labels[event_type]])), index=False, header=False)
            
            else:
                if not os.path.exists(os.path.join(src_dir, dst_dir, 'val', subject_id, labels[event_type])):
                    os.makedirs(os.path.join(src_dir, dst_dir, 'val', subject_id, labels[event_type]), exist_ok=True)
                del ts_seg['Time']
                ts_seg.to_csv(os.path.join(src_dir, dst_dir, 'val', subject_id, labels[event_type], '{:03d}.csv'.format(counts[labels[event_type]])), index=False, header=False)

            counts[labels[event_type]]+=1
            
    print(f"{subject_id} data preparation finished.")


def visualize(raw_data: dict, sfreq: float = 500.0) -> None:
    """
    Process events from raw_data, plot events using MNE, and visualize the raw data.
    
    Args:
        raw_data (dict): Dictionary obtained from reading a .mat file containing EEG data and events.
        sfreq (float): Sampling frequency, default is 500.0 Hz.
    """
    raw = mne_data(raw_data, sfreq=sfreq)
    raw.plot(duration=5, n_channels=72)  

    events = pd.DataFrame(raw_data['event']).astype({'type': int, 'init_index': int})
    events['init_time'] = (events['init_time'] * sfreq).astype(int)
    events = events.rename(columns={'init_time': 'id', 'init_index': 'test', 'type': 'event_id'})[['id', 'test', 'event_id']]
    event_dict = { 
        'led/frontside_kickturn': 11,
        'led/backside_kickturn': 12,
        'led/pumping': 13,
        'laser/frontside_kickturn': 21,
        'laser/backside_kickturn': 22,
        'laser/pumping': 23
    }
    mne.viz.plot_events(events, event_id=event_dict, sfreq=sfreq)
    plt.show()