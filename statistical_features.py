import numpy as np
import pywt
from sklearn.preprocessing import MinMaxScaler
from nolds import sampen
from scipy import stats
from EntropyHub import FuzzEn
from scipy.stats import entropy
from EntropyHub import PermEn
import antropy as ent

def mean(x):
    x = np.nan_to_num(x, nan=0)
    return np.mean(x, axis=-1)
def std(x):
    x = np.nan_to_num(x, nan=0)
    return np.std(x, axis=-1)
def ptp(x):
    x = np.nan_to_num(x, nan=0)
    return np.ptp(x, axis=-1)
def var(x):
    x = np.nan_to_num(x, nan=0)
    return np.var(x, axis=-1)
def min(x):
    x = np.nan_to_num(x, nan=0)
    return np.min(x, axis=-1)
def max(x):
    x = np.nan_to_num(x, nan=0)
    return np.max(x, axis=-1)
def argmin(x):
    x = np.nan_to_num(x, nan=0)
    return np.argmin(x, axis=-1)
def argmax(x):
    x = np.nan_to_num(x, nan=0)
    return np.argmax(x, axis=-1)
def rms(x):
    x = np.nan_to_num(x, nan=0)
    return np.sqrt(np.mean(x**2, axis=-1))
def abs_diff_signal(x):
    x = np.nan_to_num(x, nan=0)
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)
def skewness(x):
    x = np.nan_to_num(x, nan=0)
    return stats.skew(x, axis=-1)
def kurtosis(x):
    x = np.nan_to_num(x, nan=0)
    return stats.kurtosis(x, axis=-1)
def signal_energy(x):
    x = np.nan_to_num(x, nan=0)
    return np.sum(x**2, axis=-1)

def hjorth_mobility(x):
    mobilities = []
    for row in x:
        mobility = np.round(ent.hjorth_params(row, axis=-1)[0], 4)
        mobilities.append(mobility)
    return np.array(mobilities)
def hjorth_complexity(x):
    complexities = []
    for row in x:
        complexity = np.round(ent.hjorth_params(row, axis=-1)[1], 4)
        complexities.append(complexity)
    return np.array(complexities)
from sklearn.preprocessing import MinMaxScaler


def shannon_entropy(data):
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

def sample_entropy(data):
    channel_entropies = []
    for channel in data:
        m = 2
        r = 0.15 * np.std(channel)
        entropy = sampen(channel, emb_dim=m, tolerance=r)
        channel_entropies.append(entropy)
    return np.array(channel_entropies)

def approximate_entropy(data, m=2, r=0.2):
    entropies = []
    for channel in data:
        n = len(channel)
        tolerance = r * np.std(channel)
        
        def _phi(m):
            x = np.array([channel[i:i+m] for i in range(n - m + 1)])
            C = np.sum(np.all(np.abs(x[:, None] - x) <= tolerance, axis=2))
            return C / (n - m + 1)
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        if phi_m == 0 or phi_m1 == 0:
            entropies.append(float('inf'))
        else:
            ap_en = np.log(phi_m) - np.log(phi_m1)
            entropies.append(ap_en)
    
    return np.array(entropies)
# def psd_mean:

def wavelet_entropy(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    wavelet = 'db4'
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

def fuzzy_entropy(Sig, m=1, r=(0.2, 5), Fx='default'):
    scaler = MinMaxScaler()
    Sig = scaler.fit_transform(Sig)
    entropies = []
    for channel in Sig:
        channel = np.array(channel, dtype=np.float64)
        Fuzz, _, _ = FuzzEn(channel, m = 1, r = r)
        entropies.append(Fuzz)
    return np.array(entropies).flatten()

def permutational_entropy(Sig, m=4, Fx='default'):
    scaler = MinMaxScaler()
    Sig = scaler.fit_transform(Sig)
    entropies = []
    for channel in Sig:
        channel = np.array(channel, dtype=np.float64)
        perm, _, _ = PermEn(channel, m = m)
        entropies.append(perm[0])
    return np.array(entropies).flatten()

if __name__ == "__main__":
    print("Started")
    X = EH.ExampleData("gaussian")
    data = [[X], [X], [X]]
    print(shannon_entropy(data).shape)
    print(hjorth_complexity(data).shape)
    print("Ended")