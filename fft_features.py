import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from nolds import sampen
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import simpson
from tqdm import tqdm
import pywt
from numba import jit
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from EntropyHub import FuzzEn
from EntropyHub import PermEn
import antropy as ent

def get_band_powers(point, freqs, psd):
    bands = {"Delta": [0.5, 4], "Theta": [4, 8], "Alpha": [8, 12], "Beta": [12, 30], "Gamma": [30, 50]}
    band_powers = {}
    for band, (low, high), color in zip(bands.keys(), bands.values(), sns.color_palette("husl", len(bands))):
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_powers[band] = simpson(psd[idx_band], dx=freqs[1] - freqs[0])
    return band_powers
    
def get_bands_all_channels(data):
    band_powers = []
    for channel in data:
        sf = 250.
        time = np.arange(channel.size) / sf

        win =  sf
        freqs, psd = signal.welch(channel, sf, nperseg=win)

        band_power = get_band_powers(channel, freqs, psd)
        band_powers.append(band_power)
    return band_powers

def time_series_to_frequency(data):
    '''
    Data is time series data that will be converted to frequency data
    '''
    sf = 250
    time = np.arange(data.size) / sf
    freqs, psd = signal.welch(data, sf, nperseg=sf)

    return psd

def spectral_mean(data):
    psd = time_series_to_frequency(data)
    psd = np.nan_to_num(psd, nan=0)
    return np.mean(psd, axis = -1)

def spectral_std(data):
    psd = time_series_to_frequency(data)
    psd = np.nan_to_num(psd, nan=0)
    return np.std(psd, axis = -1)

def spectral_var(data):
    psd = time_series_to_frequency(data)
    psd = np.nan_to_num(psd, nan=0)
    return np.var(psd, axis = -1)

def convert_bands(band_power_dicts, band):
    features = []    
    for dictionary in band_power_dicts:
        features.append(dictionary[band])
    return features

@jit(nopython=True)
def unique_with_counts(arr):
    unique_elements = []
    counts = []
    arr_sorted = np.sort(arr)
    current_element = arr_sorted[0]
    current_count = 1
    for i in range(1, len(arr_sorted)):
        if arr_sorted[i] == current_element:
            current_count += 1
        else:
            unique_elements.append(current_element)
            counts.append(current_count)
            current_element = arr_sorted[i]
            current_count = 1
    unique_elements.append(current_element)
    counts.append(current_count)
    return np.array(unique_elements), np.array(counts)

def spectral_shannon_entropy(data):
    data = time_series_to_frequency(data)
    channel_entropies = []
    for channel in data:
        channel_entropies.append(estimate_shannon_entropy(channel))
    return np.array(channel_entropies)

def estimate_shannon_entropy(time_series_data):
    bins = 100  # Adjust the number of bins as per your data distribution
    counts, bin_edges = np.histogram(time_series_data, bins=bins)
    total_values = np.sum(counts)
    distribution = counts[counts > 0] / total_values
    entropy_value = entropy(distribution, base=2)
    return entropy_value


def wavelet_spectral_entropy(data):
    data = time_series_to_frequency(data)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    wavelet = 'db4'
    # Timing the function
    coefficients = pywt.wavedec(data, wavelet)
    def estimate_shannon_entropy(time_series_data):
        bins = 100  # Adjust the number of bins as per your data distribution
        counts, bin_edges = np.histogram(time_series_data, bins=bins)
        total_values = np.sum(counts)
        distribution = counts[counts > 0] / total_values
        entropy_value = entropy(distribution, base=2)
        return entropy_value
    
    wavelet_entropies = []
    for channel in data:
        entropy_value = estimate_shannon_entropy(coefficients[0])
        wavelet_entropies.append(entropy_value)
    
    return wavelet_entropies

def spectral_sample_entropy(data):
    sample_entropies = []
    for channel in data:
        m = 2
        r = 0.15^np.std(channel)
        entropy = sampen(channel, emb_dim=m, tolerance=r)
    psd = np.nan_to_num(psd, nan=0)

    return np.array(sample_entropies)

def spectral_fuzzy_entropy(Sig, m=1, r=(0.2, 5), Fx='default'):
    Sig = time_series_to_frequency(Sig)
    scaler = MinMaxScaler()
    Sig = scaler.fit_transform(Sig)
    entropies = []
    for channel in Sig:
        channel = np.array(channel, dtype=np.float64)
        Fuzz, _, _ = FuzzEn(channel, m = 1, r = r)
        entropies.append(Fuzz)
    return np.array(entropies).flatten()

def spectral_permutational_entropy(Sig, m=4, Fx='default'):
    Sig = time_series_to_frequency(Sig)
    scaler = MinMaxScaler()
    Sig = scaler.fit_transform(Sig)
    entropies = []
    for channel in Sig:
        channel = np.array(channel, dtype=np.float64)
        perm, _, _ = PermEn(channel, m = m)
        entropies.append(perm[0])
    return np.array(entropies).flatten()

def delta_band(data):
    deltas = []
    x = get_bands_all_channels(data)
    for channel in data:
        deltas.append(convert_bands(x, 'Delta'))
    # deltas = np.array(deltas)
    # input(deltas.shape)
    return deltas[0]

def alpha_band(data):
    alphas = []
    x = get_bands_all_channels(data)
    for channel in data:
        alphas.append(convert_bands(x, 'Alpha'))
    # alphas = np.array(alphas)
    # input(alphas.shape)
    return alphas[0]
def theta_band(data):
    thetas = []
    x = get_bands_all_channels(data)
    for channel in data:
        thetas.append(convert_bands(x, 'Theta'))
    # thetas = np.array(theta)
    # input(thetas.shape)
    return thetas[0]

def beta_band(data):
    betas = []
    x = get_bands_all_channels(data)
    for channel in data:
        betas.append(convert_bands(x, 'Beta'))
    # betas = np.array(betas)
    # input(betas.shape)
    return betas[0]

def peak_freq(data):
    time_step = 5/len(data)
    time_vect = [i*time_step for i in range(len(data))]
    sampling_freq = len(data)/5
    win = 4*sampling_freq 
    freqs, psd = signal.welch(data, sampling_freq, nperseg=win)
    peak_freq = freqs[np.argmax(psd)]
    return peak_freq

def peak_freq_channels(data):
    peak_freqs = []
    for channel in data:
        peak_freqs.append(peak_freq(channel))
    peak_freqs_array = np.array(peak_freqs)
    return peak_freqs_array

def spectral_entropy(data):
    # Replace NA values with 0
    data = np.nan_to_num(data, nan=0)
    result = ent.spectral_entropy(data, sf=250, method='welch')
    result = np.nan_to_num(result, nan=0)

    return result

def spectral_entropy_channels(data):
    entropies = []
    for channel in data:
        entropies.append(spectral_entropy(channel))
    entropies_array = np.array(entropies)
    return entropies_array

def spectral_std(data):
    time_step = 5/len(data)
    time_vect = [i*time_step for i in range(len(data))]
    sampling_freq = len(data)/5
    win = 4*sampling_freq 
    freqs, psd = signal.welch(data, sampling_freq, nperseg=win)
    return [np.std(psd), stats.kurtosis(psd)]

def spectral_std_channels(data):
    skewness = []
    for channel in data:
        skewness.append(spectral_std(channel)[0])
    skewness_array = np.array(skewness)
    return skewness_array

def spectral_kurtosis_channels(data):
    kurtosis = []
    for channel in data:
        kurtosis.append(spectral_std(channel)[1])
    kurtosis_array = np.array(kurtosis)
    return kurtosis_array


