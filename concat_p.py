from statistical_features import *
from fft_features import *

# sys.path.append("C:\\Users\\96654\\Desktop\\UROP_Training\\project1P\\data_conversion") #Change to the path of your data conversion folder
from load_data import *
import time
import numpy as np
import pandas as pd
# from bands import *
import time
import time

from joblib import Parallel, delayed
from numba import jit
import time

# Define your feature functions here (e.g., theta_band, alpha_band, etc.)
functions = {
    'theta_channels': theta_band,
    'alpha_channels': alpha_band,
    'beta_channels': beta_band,
    'delta_channels': delta_band,
    'std': std,
    'ptp': ptp,
    'var': var,
    'min': min,
    'max': max,
    'argmin': argmin,
    'argmax': argmax,
    'rms': rms,
    'abs_diff_signal': abs_diff_signal,
    'skewness': skewness,
    'kurtosis': kurtosis,
    'signal_energy': signal_energy,
    'mean': mean,
    'hjorth_complexity': hjorth_complexity,
    'hjorth_mobility': hjorth_mobility,
    'spectral_entropy_channels': spectral_entropy_channels,
    'spectral_kurtosis_channels': spectral_kurtosis_channels,
    'peak_freq_channels': peak_freq_channels,
    'spectral_skewness_channels': spectral_std_channels,
    'spectral_mean' : spectral_mean,
    # 'spectral_std' : spectral_std,
    'spectral_variance' : spectral_var,
    'sample_entropy' : sample_entropy, #TD,
    # 'shannon_entropy' : shannon_spectral_entropy, #FD
    'wavelet_entropy' : wavelet_spectral_entropy,
    'fuzzy_entropy' : fuzzy_entropy,
    'spectral_fuzzy_entropy' : spectral_fuzzy_entropy, #FD
}

def apply_functions(x, selected_functions):
    return np.array([function(x) for function in selected_functions])

def get_all_features():
    return list(functions.keys())

def concat_features(x, feature_list, save_feature_names):
    if 'all' in feature_list:
        selected_functions = list(functions.values())  # Convert to list
    else:
        selected_functions = [functions[key] for key in feature_list if key in functions]

    data = apply_functions(x, selected_functions)
    feature_names = list(functions.keys()) if 'all' in feature_list else feature_list
    concatenated_features = np.concatenate(data, axis=-1)
    if save_feature_names:
        print(f'Saving the following features: {feature_names}')
        np.save('all_feature_names.npy', feature_names)
    return concatenated_features, feature_names

def concatenate_p(data_array, feature_list, save_feature_names = False):
    start_time = time.time()
    print("Concatenating features using Parallel Processing")
    print(f"Loading the following features: {feature_list}")
    features = Parallel(n_jobs= - 1)(delayed(concat_features)(d, feature_list, save_feature_names) for d in tqdm(data_array))
    features_array = np.array([result[0] for result in features])
    end_time = time.time()
    print(f'Features Loaded in {end_time - start_time} seconds')
    return features_array

if __name__ == "__main__":
    pass