import numpy as np
import matplotlib.pyplot as plt
import spkit as sp
from spkit.data import load_data
import mne
from load_data import load_schizo
from glob import glob
import os

def plot_eeg_ATAR(X, ch_names):
    # Load EEG data
    # X, ch_names = load_data.eegSample()
    
    # Set sampling frequency
    fs = 256
    
    # Highpass filter
    Xf = sp.filter_X(X, band=[0.5], btype='highpass', fs=fs, verbose=0)
    print(Xf.shape)
    
    # Time array
    t = np.arange(Xf.shape[0]) / fs
    
    # Plot filtered EEG signal
    plt.figure(figsize=(12, 4))
    plt.plot(t, Xf + np.arange(-3, 3) * 700)  # Increased offset multiplier
    plt.xlim([t[0], t[-1]])
    plt.xlabel('Time (sec)')
    plt.yticks(np.arange(-3, 3) * 700, ch_names)  # Adjusted offset multiplier here as well
    plt.grid()
    plt.title('Xf: 14-channel EEG Signal (filtered)')
    plt.show()
    
    # Perform ICA filtering
    # XR = sp.eeg.ICA_filtering(Xf.copy(), verbose=1, winsize=128)
    
    # # Plot filtered and corrected signals
    # plt.figure(figsize=(15, 10))
    # plt.subplot(221)
    # plt.plot(t, Xf + np.arange(-3, 3) * 700)  # Increased offset multiplier
    # plt.xlim([t[0], t[-1]])
    # plt.yticks(np.arange(-3, 3) * 700, ch_names)  # Adjusted offset multiplier here as well
    # plt.grid()
    # plt.title('X: Filtered signal', fontsize=16)
    # plt.subplot(222)
    # plt.plot(t, XR + np.arange(-3, 3) * 700)  # Increased offset multiplier
    # plt.xlim([t[0], t[-1]])
    # plt.yticks(np.arange(-3, 3) * 700, ch_names)  # Adjusted offset multiplier here as well
    # plt.grid()
    # plt.title('XR: Corrected Signal', fontsize=16)
    # plt.subplot(223)
    # plt.plot(t, (Xf - XR) + np.arange(-3, 3) * 700)  # Increased offset multiplier
    # plt.xlim([t[0], t[-1]])
    # plt.xlabel('Time (s)')
    # plt.yticks(np.arange(-3, 3) * 700, ch_names)  # Adjusted offset multiplier here as well
    # plt.grid()
    # plt.title('Xf - XR: Difference \n(removed signal)', fontsize=16)
    # plt.subplot(224)
    # plt.plot(t, Xf[:, 0], label='Xf')
    # plt.plot(t, XR[:, 0], label='XR')
    # plt.xlim([t[0], t[-1]])
    # plt.xlabel('Time (s)')
    # plt.grid()
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    
    # # Compute and plot power spectrum
    # Pr1 = sp.Periodogram(Xf[:, 0])
    # Pr2 = sp.Periodogram(XR[:, 0])
    # frq = (fs / 2) * np.arange(len(Pr1)) / (len(Pr1) - 1)
    # plt.plot(frq, np.log10(Pr1), label='Xf')
    # plt.plot(frq, np.log10(Pr2), label='XR')
    # plt.legend()
    # plt.grid()
    # plt.xlim([0, frq[-1]])
    # plt.ylabel('Power Spectrum')
    # plt.xlabel('Frequency (Hz)')
    # plt.show()
    
    # Perform ATAR with different parameters
    XR1 = sp.eeg.ATAR(Xf.copy(), wv='db4', winsize=128, beta=0.0001, thr_method='ipr', OptMode='soft', verbose=1)
    XR2 = sp.eeg.ATAR(Xf.copy(), wv='db4', winsize=128, beta=0.01, thr_method='ipr', OptMode='elim')
    
    # Plot signals after ATAR correction
    plt.figure(figsize=(15, 10))
    plt.subplot(221)
    plt.plot(t, Xf + np.arange(-3, 3) * 200)
    plt.xlim([t[0], t[-1]])
    plt.yticks(np.arange(-3, 3) * 200, ch_names)
    plt.grid()
    plt.title('X: Filtered signal', fontsize=16)
    plt.subplot(222)
    plt.plot(t, XR1 + np.arange(-3, 3) * 200)
    plt.xlim([t[0], t[-1]])
    plt.yticks(np.arange(-3, 3) * 200, ch_names)
    plt.grid()
    plt.title('XR: Corrected Signal (beta=0.1)', fontsize=16)
    plt.subplot(223)
    plt.plot(t, (Xf - XR1) + np.arange(-3, 3) * 200)
    plt.xlim([t[0], t[-1]])
    plt.xlabel('Time (s)')
    plt.yticks(np.arange(-3, 3) * 200, ch_names)
    plt.grid()
    plt.title('Xf - XR: Difference \n(removed signal)', fontsize=16)
    plt.subplot(224)
    plt.plot(t, Xf[:, 0], label='Xf')
    plt.plot(t, XR1[:, 0], label='XR')
    plt.xlim([t[0], t[-1]])
    plt.xlabel('Time (s)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot power spectrum after ATAR correction
    Pr1 = sp.Periodogram(Xf[:, 0])
    Pr2 = sp.Periodogram(XR1[:, 0])
    frq = (fs / 2) * np.arange(len(Pr1)) / (len(Pr1) - 1)
    plt.plot(frq, np.log10(Pr1), label='Xf')
    plt.plot(frq, np.log10(Pr2), label='XR (beta=0.1)')
    plt.legend()
    plt.grid()
    plt.xlim([0, frq[-1]])
    plt.ylabel('Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.show()
    
    # Plot signals after ATAR correction with different beta value
    plt.figure(figsize=(15, 10))
    plt.subplot(221)
    plt.plot(t, Xf + np.arange(-3, 3) * 200)
    plt.xlim([t[0], t[-1]])
    plt.yticks(np.arange(-3, 3) * 200, ch_names)
    plt.grid()
    plt.title('X: Filtered signal', fontsize=16)
    plt.subplot(222)
    plt.plot(t, XR2 + np.arange(-3, 3) * 200)
    plt.xlim([t[0], t[-1]])
    plt.yticks(np.arange(-3, 3) * 200, ch_names)
    plt.grid()
    plt.title('XR: Corrected Signal (beta=0.01)', fontsize=16)
    plt.subplot(223)
    plt.plot(t, (Xf - XR2) + np.arange(-3, 3) * 200)
    plt.xlim([t[0], t[-1]])
    plt.xlabel('Time (s)')
    print(XR1.transpose().shape)
    return XR1.transpose() * 10 ** -7

def compute_eeg_ATAR(X, ch_names):
    # Set sampling frequency
    fs = 128
    
    # Highpass filter
    Xf = sp.filter_X(X, band=[0.5], btype='highpass', fs=fs, verbose=0)
    
    # Perform ICA filtering
    # XR = sp.eeg.ICA_filtering(Xf.copy(), verbose=0, winsize=128)
    
    # Compute power spectrum


    # Pr1 = sp.Periodogram(Xf[:, 0])
    # Pr2 = sp.Periodogram(XR[:, 0])
    
    # Perform ATAR with different parameters
    XR1 = sp.eeg.ATAR(Xf.copy(), wv='db4', winsize=128, beta=0.01, thr_method='ipr', OptMode='soft')
    # XR2 = sp.eeg.ATAR(Xf.copy(), wv='db4', winsize=128, beta=0.1, thr_method='ipr', OptMode='elim')
    
    return XR2.transpose()



def load_schizo2(file_path = "C:\\Users\\96654\\Desktop\\UROP_Training\\Project1\\Data\\Raw_Data/*.edf"):
    '''
    Purpose: Reads the data stored in the form of edf files for patients with schizophrenia in a format we can work with. 3D Numpy array: First D => Trials/Epochs : Second D => Channels : Third D => Signal
    Parameters: None
    Return Value: Data and label arrays
    Additional Data: You can download the data from here: 
    '''

    print(file_path)
    all_file_paths = glob(file_path)
    healthy_file_paths = [path for path in all_file_paths if os.path.basename(path)[0] == 'h']
    # healthy_file_paths = healthy_file_paths[:9]
    patient_file_paths = [path for path in all_file_paths if os.path.basename(path)[0] =='s']
    # patient_file_paths =   patient_file_paths[:9]

    def read_data(edf_fname):
        edf_raw = mne.io.read_raw_edf(edf_fname, preload=True)
        edf_data, times = edf_raw[:, :]
        print(edf_data.shape)
        # input()
        return edf_data

    control_epochs_array = [read_data(path) for path in healthy_file_paths]
    patient_epochs_array = [read_data(path) for path in patient_file_paths]

    control_lengths = [len(array[0]) for array in control_epochs_array]
    patient_lengths = [len(array[0]) for array in patient_epochs_array]

    print("Control Lengths:", control_lengths)
    print("Patient Lengths:", patient_lengths)
    # input()
    return control_epochs_array, patient_epochs_array


def process_data(point):
    s_ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    with open('counter.txt', 'a') as file:
        # Write "hey" to the file
        file.write("hey")
    return compute_eeg_ATAR(point.transpose() * (10**7), s_ch_names)

def clean_data(X, ch_names):
    fs = 256
    Xf = sp.filter_X(X, band=[0.5], btype='highpass', fs=fs, verbose=0)
    print(Xf.shape)
    t = np.arange(Xf.shape[0]) / fs
    XR1 = sp.eeg.ATAR(Xf.copy(), wv='db4', winsize=128, beta=0.0001, thr_method='ipr', OptMode='soft', verbose=1)
    return XR1.transpose() * 10 ** -7
if __name__ == "__main__":
    data_array = np.load('ASD_data_array1.npy')
    print(data_array.shape)
    ASD_channels = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
    # plot_eeg_ATAR(data_array[0].transpose() * 10**7, ASD_channels)

    clean_data_array = [clean_data(sample.transpose() * 10 ** 7, ASD_channels) for sample in data_array]
    clean_data_array = np.array(clean_data_array)
    print(clean_data_array.shape)
    np.save('ASD_clean_data_array1.npy', clean_data_array)

    input()
    cleaning()
    main()

    
