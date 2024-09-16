import csv
from glob import glob
import os
import mne
import numpy as np
from tqdm import tqdm
import time
import pickle
import re
from pathlib import Path



def load_alc(path):
    '''
    Purpose: Converts the data stored in alcohol data in csv files into a format we can work with. 3D Numpy array: First D => Trials/Epochs : Second D => Channels : Third D => Signal
    Parameters: None
    Returns: The data array and label array (Numpy)
    Additional Data: You can download the data from here: 
    '''
    paths = glob(path)
    # print(paths)
    # Open the CSV file
    temp_list = []
    trials = []
    chan_names = ['FP1','FPZ','AFZ','AF1','FZ','FCZ','FC4','C4','CPZ','CP4','P5','PZ','P1','PO2','FPZ','OZ']
    print("Loading Data")
    for path in tqdm(paths):
        channel = []
        with open(path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                temp_list.append(row)
                if row[9] == '0.0':
                    del temp_list[-1]
                    channel.append(temp_list)
                    temp_list =  []
                    temp_list.append(row)
        trials.append(channel)
        temp_list = []
        del channel[0]
        channel = []
    label_array = []
    for trial in trials:
        label_array.append((trial[0][0][5]))

    for trial in trials:
        for channel in trial:
            for i, value in enumerate(channel):
                channel[i] = float(value[4])

    print(trials[0][0][0])
    data_array = np.array(trials)
    label_array = np.array(label_array)

    return trials, label_array

def load_ASD(file_path="Data/ASD_Data/*.set", augmented = False):
    def read_data(path, show = False, augmented = False):
        try:
            imp_channel = ['PO4', 'Oz', 'Fp1', 'P10', 'P8', 'O2', 'C3', 'F8', 'P1', 'Fpz']
            raw = mne.io.read_raw_eeglab(path, preload = True)
            channels = raw.info.ch_names
            remove_channels = []
            for channel in channels:
                if channel not in imp_channel:
                    remove_channels.append(channel)
            raw.info['bads'] = remove_channels
            print(raw)
            print(raw.info)
            if len(remove_channels) != 0:
                raw.drop_channels(remove_channels)
            if show:
                raw.plot(block = True)
            raw.set_eeg_reference()
            raw.filter(l_freq=0, h_freq=30)
            if augmented:
                overlap = 2
            else:
                overlap = 1
            epochs = mne.make_fixed_length_epochs(raw, duration=5, overlap=overlap)
            array = epochs.get_data()
            print(array.shape)
            if array.shape[1] == 10:
                print(raw.info)
                print(array.shape[1])
                # input()
                return array
            else:
                raise ValueError
        except ValueError:
            # input("Error ")
            return 1
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, file_path)  

    paths = glob(full_path)
    healthy_file_paths = []
    ASD_file_paths = []
    for path in paths:
        base_dir = Path(base_dir)
        path = Path(path)
        relative_path = path.relative_to(base_dir)
        sample_name = str(relative_path).split('ASD_Data\\')[1]
        if int(sample_name[0] + sample_name[1]) <= 28:
            ASD_file_paths.append(path)
        else:
            healthy_file_paths.append(path)

    patient_epochs_array = [read_data(path, show = False, augmented = augmented) for path in ASD_file_paths[:len(ASD_file_paths) - 1] if type(read_data(path, show = False, augmented = augmented)) != int]
    control_epochs_array = [read_data(path, show = False, augmented = augmented) for path in healthy_file_paths if type(read_data(path, show = False, augmented = augmented)) != int]

    patient_epochs_array = np.concatenate(patient_epochs_array, axis=0)
    control_epochs_array = np.concatenate(control_epochs_array, axis=0)

    if augmented:
        control_epochs_labels = [624*[0]]
        patient_epochs_labels = [936*[1]]
    else:
        control_epochs_labels = [468*[0]]
        patient_epochs_labels = [702*[1]]

    print(len(control_epochs_array))
    print(len(patient_epochs_array))


    channel_lens = []
    for epoch in patient_epochs_array:
        channel_lens.append(len(epoch))
    print(channel_lens)
    label_list = control_epochs_labels + patient_epochs_labels
    print(np.array(control_epochs_array).shape)
    print(np.array(patient_epochs_array).shape)
    data_array = np.vstack((control_epochs_array, patient_epochs_array))
    label_array = np.hstack(label_list)
   
    return data_array, label_array


def load_schizo(file_path="Data/Schizophrenia_Data/*.edf"):
    '''
    Purpose: Reads the data stored in the form of edf files for patients with schizophrenia in a format we can work with. 3D Numpy array: First D => Trials/Epochs : Second D => Channels : Third D => Signal
    Parameters: None
    Return Value: Data and label arrays
    Additional Data: You can download the data from here: 
    '''
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, file_path)

    print(file_path)
    all_file_paths = glob(full_path)
    healthy_file_paths = [path for path in all_file_paths if os.path.basename(path)[0] == 'h']
    # healthy_file_paths = healthy_file_paths[:9]
    patient_file_paths = [path for path in all_file_paths if os.path.basename(path)[0] == 's']
    # patient_file_paths =   patient_file_paths[:9]

    def read_data(file_path):
        data = mne.io.read_raw_edf(file_path, preload=True)
        data.set_eeg_reference()
        data.filter(l_freq=0.5, h_freq=30)
        epochs = mne.make_fixed_length_epochs(data, duration=5, overlap=2)  # Duration 5s => timestep = values/5
        array = epochs.get_data()
        print(data.info)
        cur_seconds = len(array) * 4
        print(data.info.ch_names)
        print(f"Total time {cur_seconds}")
        # input()
        return [array, cur_seconds]


    # Data Organization
    control_epochs_array = [read_data(path)[0] for path in healthy_file_paths]
    patient_epochs_array = [read_data(path)[0] for path in patient_file_paths]

    print("Calculating average time :")
    healthy_total_time = 0
    schizo_total_time = 0
    for i in range(len(healthy_file_paths)):
        cur_sec1 = read_data(healthy_file_paths[i])[1]
        cur_sec2 = read_data(patient_file_paths[i])[1]
        healthy_total_time += cur_sec1 
        schizo_total_time += cur_sec2
    print(f"Healthy Total Time is {healthy_total_time} seconds")
    print(f"Healthy Average Time is {healthy_total_time/14}")
    print(f"Schizo Total Time is {schizo_total_time} seconds")
    print(f"Schizo Average Time is {schizo_total_time/14}")
    # input()
    total_epochs = sum(data.shape[0] for data in control_epochs_array) + sum(data.shape[0] for data in patient_epochs_array)

    control_epochs_labels = [len(i)*[0] for i in control_epochs_array]
    patient_epochs_labels = [len(i)*[1] for i in patient_epochs_array]

    data_list = control_epochs_array + patient_epochs_array
    label_list = control_epochs_labels + patient_epochs_labels

    group_list = [[i]*len(j) for i,j in enumerate(data_list)]

    data_array = np.vstack(data_list)
    label_array = np.hstack(label_list)
    group_array = np.hstack(group_list)
    label1 = sum([i for i in label_array if i == 0])
    label2 = sum([i for i in label_array if i == 1])
    print(f"Control size is {label1}")
    print(f"Schizo size is {label2}")

    print(f"Data array shape: {data_array.shape}")
    print(f"Label array shape: {label_array.shape}")
    # input()
    return data_array, label_array

if __name__ =="__main__":
    def main():
        # data_array, label_array = load_ASD1()
        # print(data_array.shape)
        # print(label_array.shape)
        # np.save('ASD_data_array1.npy' , data_array)
        # np.save('ASD_label_array1.npy' , label_array)
        data_array = np.load('ASD_data_array1.npy')
        label_array = np.load('ASD_label_array1.npy')
        print(data_array.shape)
        print(label_array.shape)

    # print(np.load('ASD_data_array2.npy').shape)
    # input()
    data_array, label_array = load_schizo()
    np.save('npy_arrays/s_data_array.npy', data_array)
    np.save('npy_arrays/s_label_array.npy', label_array)
    print("Done")
    input()
