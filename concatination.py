from statistical_features import *
from fft_features import *
import sys
import random
# sys.path.append("C:\\Users\\96654\\Desktop\\UROP_Training\\project1P\\data_conversion") #Change to the path of your data conversion folder
from load_data import *
import time
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import time

def get_random_elements(arr):
    n = random.randint(1, 2)  # Choose how many elements to select (1 to 3)
    return random.sample(arr, n)

def concatinate_random(data_array):
    '''
    Purpose: Gets the desired feature array from a data_array. The functions diuctionary contains several features, we can choose which ones to run and try to obtain the best scores.
    Parameters: Data Array 
    Return Value: Feature array
    '''

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
        'rms': rms,
        'abs_diff_signal': abs_diff_signal,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'signal_energy': signal_energy,
        'mean': mean,
        'hjorth_complexity': hjorth_complexity,
        'hjorth_mobility': hjorth_mobility,
        'spectral_kurtosis_channels': spectral_kurtosis_channels,
        'peak_freq_channels': peak_freq_channels,
        'spectral_skewness_channels': spectral_std_channels,
        'spectral_mean': spectral_mean,
        'spectral_std': spectral_std,
        'spectral_variance': spectral_var,
        'shannon_entropy' : shannon_entropy,
        'spectral_shannon_entropy': spectral_shannon_entropy, # FD
        'wavelet_entropy': wavelet_entropy, #
        'spectral_wavelet_entropy' : wavelet_spectral_entropy,
        # 'sample_entropy' : sample_entropy,
        # 'spectral_sample_entropy' : spectral_sample_entropy,
        'fuzzy_entropy' : fuzzy_entropy,
        'spectral_fuzzy_entropy' : spectral_fuzzy_entropy,
        # 'permutation_entropy' : permutational_entropy,
        # 'spectral_permutation_entropy' : spectral_permutational_entropy
    }
    features = []
    random_keys = random.sample(list(functions.keys()), random.randint(3, 6))
    print(random_keys)
    # time.sleep(5)
    selected_values = [functions[key] for key in random_keys]
    for d in tqdm(data_array):
        feature_concatenated = np.concatenate([func(d) for func in selected_values], axis=-1)
        features.append(feature_concatenated)

    features_array = np.array(features)
    
    return features_array, selected_values




def concatenate(data_array, feature_list):
    '''
    Purpose: Concatenates desired feature arrays from a data_array based on a feature_list.
    Parameters:
    - data_array: Array of data where each element contains features to concatenate.
    - feature_list: List of features to concatenate. Can contain specific features or 'all' to select all features.
    Return Value:
    - features_array: Concatenated feature array.
    '''
    
    def concat_features(x, feature_list):
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
            'spectral_std' : spectral_std,
            'spectral_variance' : spectral_var,
            # 'sample_entropy' : sample_entropy, #TD,
            # 'wavelet_entropy' : wavelet_spectral_entropy,
            # 'fuzzy_entropy' : fuzzy_entropy,
            # 'spectral_fuzzy_entropy' : spectral_fuzzy_entropy, #FD
        }
        
        if 'all' in feature_list:
            selected_functions = functions
        else:
            selected_functions = {key: value for key, value in functions.items() if key in feature_list}
        
        data = {}
        for function_name, function in selected_functions.items():
            data[function_name] = function(x)
        
        concatenated_features = np.concatenate(list(data.values()), axis=-1)
        feature_names = list(data.keys())
        return concatenated_features, feature_names

    features = []
    print("Concatenating features...")
    for d in tqdm(data_array):
        concatenated_features, _ = concat_features(d, feature_list)
        features.append(concatenated_features)
    
    features_array = np.array(features)
    return features_array


if __name__ == "__main__":
    pass