def append_directories():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths to the folders
    folders = [
        'ASD_Data',
        'csv_and_txt_files',
        'data_loading',
        'feature_engineering_code',
        'model_creation',
        'npy_arrays'
    ]

    for folder in folders:
        folder_path = os.path.join(current_dir, folder)
        sys.path.append(folder_path)
        print(folder_path)
        time.sleep(2)

import multiprocessing
import os
import json
import re
from concatination import *
# from pushbullet import Pushbullet
from model import bagging_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV, RandomizedSearchCV, LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from matplotlib.animation import FuncAnimation
import os
import sys
import pyttsx3
from scipy.signal import convolve
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix

def say(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    engine.say(text)
    engine.runAndWait()
def graph_frac_acc(fractions, accuracies):
    fractions_np = np.array(fractions)
    accuracies_np = np.array(accuracies)
    plt.figure(figsize=(8, 6))
    plt.plot(fractions_np, accuracies_np, marker='o', linestyle='-', color='b', label='Data Line')
    plt.xlabel('Fraction of Sure to All')
    plt.ylabel('Accuracy of Sure Samples')
    plt.title('Fraction vs Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_eeg_data(eeg_data):
    """
    Plots EEG data with shape (channels, points) in separate subplots for each channel.

    Parameters:
    eeg_data (np.ndarray): 2D array with shape (channels, points), where
                           'channels' is the number of EEG channels and
                           'points' is the number of data points in each channel.
    """
    # Ensure the input is a numpy array
    eeg_data = np.array(eeg_data)
    
    # Check shape of the input data
    if eeg_data.ndim != 2:
        raise ValueError("Input data must be a 2D array with shape (channels, points).")
    
    channels, points = eeg_data.shape
    
    # Create a time vector (assuming sampling rate is 1 Hz for simplicity)
    time = np.arange(points)
    
    # Create subplots
    fig, axes = plt.subplots(channels, 1, figsize=(12, 2 * channels), sharex=True, sharey=True)
    
    if channels == 1:
        axes = [axes]  # Make sure axes is iterable if there is only one channel
    
    # Plot each channel in its own subplot
    for i in range(channels):
        axes[i].plot(time, eeg_data[i], label=f'Channel {i+1}')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend()
        axes[i].grid(True)
    
    # Add labels and title
    axes[-1].set_xlabel('Time')
    plt.suptitle('EEG Data', y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show plot

def percentage_of_most_occuring_element(arr):
    from collections import Counter
    counts = Counter(arr)
    most_common_element, most_common_count = counts.most_common(1)[0]
    total_elements = len(arr)
    
    # Calculate the percentage
    percentage = (most_common_count / total_elements) * 100
    
    return percentage

def feature_bagging_loader(data_array, feature_lists, saved):
    from concat_p import concatenate_p
    all_feature_arrays = []

    for i, feature_lst in enumerate(feature_lists):
        path = f'bagging_npy//array{i}.npy'
        if saved:
            cur_feature_array = np.load(path)
            all_feature_arrays.append(cur_feature_array)
        else:
            cur_feature_array = concatenate_p(data_array, feature_lst)
            np.save(path, cur_feature_array)
    print(f'{i} Arrays Loaded')

    return all_feature_arrays

def more_probable_class(*preds):
    preds = [np.array(pred) for pred in preds]
    
    classes = [np.argmax(pred) for pred in preds]
    probabilities = [pred[cls] for pred, cls in zip(preds, classes)]
    
    max_prob_index = np.argmax(probabilities)
    
    return classes[max_prob_index]
    
def plot_calibration_curve(y_true, y_prob, title='Calibration Curve'):
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_time_series(time_series_data, title="Time Series Data", xlabel="Time", ylabel="Value"):
    """
    Plots time series data.

    Parameters:
    time_series_data (pd.Series or pd.DataFrame): Time series data with a datetime index.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(time_series_data, linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

def calculate_diversity_score(feature_lists):
    n = len(feature_lists)
    diversity_scores = np.zeros(n)
    
    # Compute pairwise Jaccard similarity and average it
    for i in range(n):
        for j in range(i + 1, n):
            set_i = set(feature_lists[i])
            set_j = set(feature_lists[j])
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            jaccard_index = intersection / union if union != 0 else 0
            diversity_scores[i] += jaccard_index
            diversity_scores[j] += jaccard_index
    
    # Normalize scores
    diversity_scores = 1 - (diversity_scores / (n - 1))
    return diversity_scores

def filter_feature_lists(feature_dict, threshold=0.5):
    feature_lists = list(feature_dict.values())
    
    diversity_scores = calculate_diversity_score(feature_lists)
    
    filtered_dict = {}
    for score, (accuracy, features) in zip(diversity_scores, feature_dict.items()):
        if score >= threshold:
            filtered_dict[accuracy] = features
    
    return filtered_dict

def feature_bagging_helper(accuracy_threshold = 0.95, diversity_threshold = 0.75):
    with open('sole_models_random.txt', 'r') as file:
        lines = file.readlines()
    accuracies = []
    feature_lists = []
    models = []
    for line in lines:
        match = re.search(r'Accuracy:\s*([\d.]+)', line)
        feature_list_pattern = r"Feature List:\s*(\[[^\]]*\])"
        feature_list_match = re.search(feature_list_pattern, line)
        feature_list = eval(feature_list_match.group(1)) if feature_list_match else []
        model_pattern = r"Model:\s*([^\s]+)"
        model_match = re.search(model_pattern, line)
        model = model_match.group(1) if model_match else None
        if match:
            accuracy = float(match.group(1))
            if accuracy > accuracy_threshold:
                feature_lists.append(feature_list)
                accuracies.append(accuracy)
                models.append(model)
    print(len(feature_lists))
    print(len(models))
    f = {}
    for feature_list, accuracy in zip(feature_lists, accuracies):
        f[accuracy] = feature_list

    sorted_dict = dict(sorted(f.items(), key=lambda item: item[0], reverse=True))
    filtered_dict = filter_feature_lists(sorted_dict, diversity_threshold)
    filtered_features = filtered_dict.values()
    return feature_lists, models, accuracies, filtered_features

def prepend_text_to_file_in_place(file_path, text_to_prepend):
    try:
        print(f"Reading from file: {file_path}")
        with open(file_path, 'r') as file:
            lines = file.readlines()
        print(f"Number of lines read: {len(lines)}")
        with open(file_path, 'w') as file:
            for line in lines:
                new_line = f"{text_to_prepend}{line}"
                file.write(new_line)
        print(f"Successfully prepended text to each line in {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

def organize_csv(file_path):
    output_file_path = file_path
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    parsed_lines = []
    for line in lines:
        feature_list_start = line.index('[')
        feature_list_end = line.index(']')
        accuracy_start = line.index('Accuracy: ')
        features = line[feature_list_start + 1:feature_list_end].split(', ')
        accuracy = float(line[accuracy_start + len('Accuracy: '):])
        parsed_lines.append((features, accuracy))
        sorted_lines = sorted(parsed_lines, key=lambda x: x[1], reverse=True)
        with open(output_file_path, 'w') as output_file:
            for features, accuracy in sorted_lines:
                feature_str = ', '.join(features)
                output_file.write(f'[{feature_str}] Accuracy: {accuracy}\n')

import ast

def string_to_list(s):
    try:
        # Use ast.literal_eval to safely evaluate the string representation of the list
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
        else:
            raise ValueError("The string does not represent a list.")
    except (SyntaxError, ValueError) as e:
        print(f"Error: {e}")
        return None

def schizo_load():
    feature_array, functions = concatinate_random_features(self.feature_array)
    model, x, y, accuracy = voting_model2(feature_array, self.label_array)
    txt_file = "features2.txt"
    function_names = [func.__name__ for func in functions]
    # # from pushbullet import Pushbullet
    key = "o.AQmxuEOjusfB36tjOd8fBJ3bvzLyGUUk"
    # pb = Pushbullet(key)
    push = pb.push_note("Accuracy Update","The current accuracy is {}%".format(accuracy))
    with open(txt_file, mode='a') as file:
        file.write("Feature List: " + str(function_names) +
                    "   Accuracy: " + str(accuracy) + "\n")
    organize_csv('features2.txt')

def alc_load():
    feature_array, functions = concatinate_random_features(self.feature_array)
    model, x, y, accuracy = voting_model(feature_array, self.label_array)
    txt_file = "features.txt"
    function_names = [func.__name__ for func in functions]
    from pushbullet import Pushbullet
    key = "o.AQmxuEOjusfB36tjOd8fBJ3bvzLyGUUk"
    pb = Pushbullet(key)
    push = pb.push_note("Accuracy Update","The current accuracy is {}%".format(accuracy))
    with open(txt_file, mode='a') as file:
        file.write("Feature List: " + str(function_names) +
                    "   Accuracy: " + str(accuracy) + "\n")
    organize_csv('features.txt')
    
def get_best_features(txt_file_name):
    txt_file_name = 'csv_and_txt_files\\' + txt_file_name
    with open(txt_file_name, 'r') as f:
        string= f.readlines()[0]
    pattern = r"Feature List: \[([^\[\]]+)\]"
    match = re.search(pattern, string)
    if match:
        feature_list_str = match.group(1)
        feature_list = [item.strip("' ") for item in feature_list_str.split(',')]
    else:
        print("No feature list found in the string.")
    return feature_list
def train_and_evaluate(features, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    clf = ExtraTreesClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def write_to_file(filename, text):
    with open(filename, 'a') as file:
        output = text + '.npy' + '\n'
        file.write(output)

def read_npy_arrays():
    with open('csv_and_txt_files/feature_array_names.txt', 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        print(f'{i}: {line}')


def create_test_array(predictions, similarity=0.95):
    predictions = np.array(predictions)
    num_elements_to_change = int(len(predictions) * (1 - similarity))
    test_array = np.copy(predictions)
    indices_to_change = np.random.choice(len(predictions), num_elements_to_change, replace=False)
    test_array[indices_to_change] = 1 - test_array[indices_to_change]
    
    return test_array

def get_random_feature_lists(number):
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
    # 'fuzzy_entropy' : fuzzy_entropy,
    # 'spectral_fuzzy_entropy' : spectral_fuzzy_entropy,
    # 'permutation_entropy' : permutational_entropy,
    # 'spectral_permutation_entropy' : spectral_permutational_entropy
    }
    feature_names = functions.keys()
    all_random_features = []
    for i in range(number):
        random_names = random.sample(list(feature_names), random.randint(2, 7))
        all_random_features.append(random_names)
    return all_random_features

def random_features_helper_schizo(data_array, label_array):
    from model import sole_models
    print("Starting: Number of CPU cores" + str(multiprocessing.cpu_count()))
    feature_array, functions = concatinate_random(data_array)
    accuracy_dict = sole_models(feature_array, label_array, test_size = 0.1)
    accuracy_dict = dict(sorted(accuracy_dict.items(), key=lambda item: item[1], reverse=True))
    model = next(iter(accuracy_dict))
    accuracy = accuracy_dict[model]
    txt_file = "sole_models_random.txt"
    function_names = [func.__name__ for func in functions]
    with open(txt_file, mode='a') as file:
        file.write("Feature List: " + str(function_names) + "   Accuracy: " + str(accuracy) + "   Model: " + str(model) + "\n")
    print("Models Ran successfully")

def random_features_helper_alc(data_array, label_array):
    print("Starting: Number of CPU cores" + str(multiprocessing.cpu_count()))
    feature_array, functions = concatinate_random(data_array)
    model, x, y, accuracy = voting_model(feature_array, label_array)
    txt_file = "features.txt"
    function_names = [func.__name__ for func in functions]
    key = "o.AQmxuEOjusfB36tjOd8fBJ3bvzLyGUUk"
    pb = Pushbullet(key)
    push = pb.push_note("Accuracy Update", "The current accuracy is {}%".format(accuracy))
    with open(txt_file, mode='a') as file:
        file.write("Feature List: " + str(function_names) + "   Accuracy: " + str(accuracy) + "\n")
    organize_csv('features2.txt')

def random_features_helper_ASD(data_array, label_array, clean):
    organize_csv('features6.txt')
    print("Starting: Number of CPU cores" + str(multiprocessing.cpu_count()))
    feature_array, functions = concatinate_random2(data_array)
    accuracy = voting_model(feature_array, label_array)
    if clean:
        txt_file = "features4.txt"
        # input("Good")
    else:
        txt_file = "features6.txt"
    function_names = functions
    key = "o.AQmxuEOjusfB36tjOd8fBJ3bvzLyGUUk"
    pb = Pushbullet(key)
    push = pb.push_note("Accuracy Update", "The current accuracy is {}%".format(accuracy))

    with open(txt_file, mode='a') as file:
        file.write("Feature List: " + str(function_names) + "   Accuracy: " + str(accuracy) + "\n")
    organize_csv('features6.txt')

def test_svm(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels)
    from svm import SVM_clasifier
    model = SVM_clasifier(learning_rate = 0.000001, kernel = "linear", n_iterations = 1000, lambda_parameter = 0.01, sigma = 3)
    model.fit(X_train, y_train)
    pred1 = model.predict(X_train)
    print(pred1)
    print(y_train)
    input()
    training_accuracy = accuracy_score(y_train.astype(int), pred1)
    print("Training Accuracy:", training_accuracy)

def print_metrics(report, model_name):
    accuracy = report["accuracy"]
    print()
    green_color = '\033[92m'
    reset_color = '\033[0m'
    print(f"{green_color}{model_name} Accuracy: {accuracy * 100:.3f}%{reset_color}")
    data = {
        "Metric": ["Precision", "Recall", "F1-Score", "Support"],
        "Patient": [report['0']['precision'], report['0']['recall'], report['0']['f1-score'], report['0']['support']],
        "Control": [report['1']['precision'], report['1']['recall'], report['1']['f1-score'], report['1']['support']],
        "Macro Avg": [report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score'], report['macro avg']['support']],
        "Weighted Avg": [report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score'], report['weighted avg']['support']]
    }

    df = pd.DataFrame(data)
    df.set_index('Metric', inplace=True)

    # Display the DataFrame in a well-formatted way
    print(df)



def graph_learning_curve(feature_array, label_array, increments):
    accuracies = []
    test_sizes = np.arange(increments, 1.0, increments)
    print("Trying the following test sizes: " + str(test_sizes))
    
    for i in test_sizes:
        # cur_accuracy, _, _, _, _ = voting_model(feature_array, label_array, test_size =  i)
        _, _, cur_accuracy = bagging_model(feature_array, label_array, test_size=i, model='svm', n_estimators=5)
        accuracies.append(cur_accuracy)
    
    plt.plot(test_sizes, accuracies, marker='o')
    plt.title('Learning Curve')
    plt.xlabel('Test Set Size')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.0) 
    plt.grid(True)
    plt.show()
    
    return accuracies


def graph_PCA(feature_array, label_array, n_components):
    scaler = StandardScaler()
    scaler.fit(feature_array)
    scaled_features = scaler.transform(feature_array)
    num_frames = 1000
    gamma_values = np.linspace(10**3, (1 + 10**-15)* 10**3, num_frames)
    
    kpca_results = []
    print("Loading PCA for gamma values")
    for gamma_value in tqdm(gamma_values):
        kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma_value)
        x_kpca = kpca.fit_transform(scaled_features)
        kpca_results.append(x_kpca)
    
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter([], [], cmap='viridis')
        ax.set_xlabel('First principal component')
        ax.set_ylabel('Second principal component')
        ax.set_title('Kernel PCA with RBF Kernel')

    elif n_components == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter([], [], [], cmap='viridis')
        ax.set_xlabel('First principal component')
        ax.set_ylabel('Second principal component')
        ax.set_zlabel('Third principal component')
        ax.set_title('Kernel PCA with RBF Kernel')

    else:
        raise ValueError("Number of components must be 2 or 3.")
    
    def update_plot(i):
        x_kpca = kpca_results[i]
        if n_components == 2:
            scatter.set_offsets(x_kpca[:, :2])
            scatter.set_array(label_array)  # Set color array
            ax.set_xlim(x_kpca[:, 0].min(), x_kpca[:, 0].max())
            ax.set_ylim(x_kpca[:, 1].min(), x_kpca[:, 1].max())
            ax.set_title(f'Kernel PCA with RBF Kernel (Gamma={gamma_values[i]:.2e})')
        
        elif n_components == 3:
            scatter._offsets3d = (x_kpca[:, 0], x_kpca[:, 1], x_kpca[:, 2])
            scatter.set_array(label_array)  # Set color array
            ax.set_xlim(x_kpca[:, 0].min(), x_kpca[:, 0].max())
            ax.set_ylim(x_kpca[:, 1].min(), x_kpca[:, 1].max())
            ax.set_zlim(x_kpca[:, 2].min(), x_kpca[:, 2].max())
            ax.set_title(f'Kernel PCA with RBF Kernel (Gamma={gamma_values[i]:.2e})')

    anim = FuncAnimation(fig, update_plot, frames=num_frames, interval=10)
    plt.show()

def SHAP_ROC_AUC_Threshold(features_array, label_array, features, channels):
    df = pd.DataFrame()
    for i in range(features_array.shape[1]):
        column_name = f'Column{i+1}' 
        df[column_name] = features_array[:, i]
    features_array = df
    label_array = pd.DataFrame({'Diagnosis': label_array})

    X_train, X_test, y_train, y_test = train_test_split(features_array, label_array, test_size=0.1)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    shap.initjs()
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)

    # Debugging: Check the shape of shap_values and the expected feature names length
    print(f"shap_values shape: {shap_values.shape}")
    num_features = shap_values.shape[1]
    feature_names = [f"Channel {channel} - {feature}" for channel in channels for feature in features]
    print(f"Number of features in shap_values: {num_features}")
    print(f"Number of feature names: {len(feature_names)}")
    
    if len(feature_names) != num_features:
        raise ValueError("The feature_names array length does not match the number of features in shap_values")

    # Plotting SHAP values
    shap.plots.force(explainer.expected_value[0], shap_values=shap_values[0][0], features=X_test.iloc[0, :], feature_names=feature_names, matplotlib=True)
    shap.decision_plot(explainer.expected_value[0], shap_values[1][12], features=X_test, feature_names=feature_names)
    shap_fig = shap.summary_plot(shap_values=shap_values[0], features=X_test, feature_names=feature_names, show=False)
    display(shap_fig)


def append_directories():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths to the folders
    folders = [
        'ASD_Data',
        'csv_and_txt_files',
        'npy_arrays'
    ]

    for folder in folders:
        folder_path = os.path.join(current_dir, folder)
        sys.path.append(folder_path)
        # print(folder_path)
def top_features_by_variance(features, top_n=100):
    # Calculate variance along axis 0 (features)
    variances = np.var(features, axis=0)
    
    # Get indices of top-n features with highest variance
    top_indices = np.argsort(variances)[-top_n:][::-1]
    
    # Retrieve the top-n features
    top_features = features[:, top_indices]
    
    return top_features, top_indices, variances[top_indices]

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def graph_features(channels, feature_names, all_feature_array):
    # Initialize lists to store selected features
    selected_features = []
    
    # Split each feature in all_feature_array and store in selected_features
    for x in all_feature_array:
        temp_array = x.split(' ')
        selected_features.append(temp_array)
    
    # Count occurrences of each channel and feature
    channel_counts = {channel: 0 for channel in channels}
    feature_counts = {feature: 0 for feature in feature_names}
    
    for feature in selected_features:
        channel, feature_name = feature
        channel_counts[channel] += 1
        feature_counts[feature_name] += 1
    
    # Sort channels by the number of features
    sorted_channels = sorted(channels, key=lambda x: channel_counts[x], reverse=True)
    
    # Sort feature names by the number of occurrences
    sorted_feature_names = sorted(feature_names, key=lambda x: feature_counts[x], reverse=True)
    
    # Create a mapping of channel names to indices
    channel_indices = {channel: idx for idx, channel in enumerate(sorted_channels)}
    
    # Create a mapping of feature names to indices
    feature_indices = {feature: idx for idx, feature in enumerate(sorted_feature_names)}
    
    # Create arrays for plotting
    x_points = []
    y_points = []
    
    for feature in selected_features:
        channel, feature_name = feature
        if channel in channel_indices and feature_name in feature_indices:  # Ensure channel and feature are in sorted lists
            x_points.append(channel_indices[channel])
            y_points.append(feature_indices[feature_name])

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(x_points, y_points, c='blue', marker='o', alpha=0.6, label='Selected Features')
    plt.xticks(range(len(sorted_channels)), sorted_channels, rotation=90)
    plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
    plt.xlabel('EEG Channels', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Selected Features for EEG Channels', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def get_proba(pred_array):
    ones = 0
    zeros = 0
    for pred in pred_array:
        if pred == 0:
            zeros += 1
        elif pred == 1:
            ones += 1
    proba = [zeros / (zeros + ones) , ones / (zeros + ones)]
    return proba
def calculate_average_proba (proba1, proba2):
    average_prob = [(proba1[0] + proba2[0]) / 2, proba1[1] + proba2[1] / 2]
    if average_prob[0] > average_prob[1]:
        return 0
    else:
        return 1
def graph_learning_curves(data):
    x = list(data.keys())
    models = next(iter(data.values())).keys()  # Get the model names from the first entry

    colors = plt.get_cmap('tab20').colors

    plt.figure(figsize=(12, 8))

    for i, model in enumerate(models):
        y = [data[x_val][model] for x_val in x]
        plt.plot(x, y, label=model, color=colors[i % len(colors)], linewidth=2)  # Set linewidth to 2

    plt.xlabel('Value')
    plt.ylabel('Performance Metric')
    plt.title('Model Performance Metrics at Different Values')
    plt.legend()
    plt.ylim(0.5, 1)
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def noise_helper(eeg_data, mean=0, noise_fraction=0.5, filter_size=10, artifact_fraction=0.01, artifact_duration=10, max_artifact_magnitude=5):
    data_std = np.std(eeg_data)
    
    noise_std = noise_fraction * data_std
    noise = np.random.normal(mean, noise_std, eeg_data.shape)
    
    filter_kernel = np.ones(filter_size) / filter_size
    smoothed_noise = convolve(noise, filter_kernel, mode='same')
    
    num_samples = eeg_data.shape[0]
    artifact_data = np.zeros(num_samples)
    
    num_artifacts = int(num_samples * artifact_fraction)
    for _ in range(num_artifacts):
        start_index = np.random.randint(0, num_samples - artifact_duration)
        magnitude = np.random.uniform(1, max_artifact_magnitude) * noise_std
        artifact_data[start_index:start_index + artifact_duration] += np.random.uniform(-1, 1) * magnitude

    noisy_eeg_data = eeg_data + smoothed_noise + artifact_data
    
    return noisy_eeg_data

def add_gaussian_noise_to_eeg(data):
    result = []
    for sample in data:
        temp = []
        for channel in sample:
            temp.append(noise_helper(channel))
        result.append(temp)
    return np.array(result)

def append_to_json(file_path, new_dict):
    try:
        # Read existing data from the file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Ensure data is a list to append new_dict
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list at the top level")

        # Append the new dictionary to the list
        data.append(new_dict)

        # Write the updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    except FileNotFoundError:
        # If the file does not exist, create it with the new_dict as the initial content
        with open(file_path, 'w') as file:
            json.dump([new_dict], file, indent=4)
    except json.JSONDecodeError:
        raise ValueError("Error decoding JSON from the file")
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
    
def save_learning_curve(dictionary, name, note, file_path='learning_curves.json'):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f) 
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    data[name] = {
        'note': note,
        'dictionary': dictionary
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Dictionary saved with name: '{name}' and note: '{note}'")

def retrieve_learning_curve(name, file_path='learning_curves.json'):
    if not os.path.exists(file_path):
        print("File does not exist.")
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    item = data.get(name, None)
    
    if item is None:
        print(f"No dictionary found for name: '{name}'")
    else:
        print(f"Dictionary retrieved for name: '{name}'")
        print(f"Note: {item['note']}")
    
    return item

def graph_feature_types_reults(results):
    # Extracting feature types and classifiers
    feature_types = list(results.keys())
    classifiers = list(results['Band and Signal Power'].keys())

    # Setting up bar positions and width
    bar_width = 0.05
    index = np.arange(len(feature_types))
    opacity = 0.6

    # Plotting each classifier's results
    colors = ['g', 'c', 'b', 'm', 'y', 'k', 'purple', 'blue', 'cyan', 'turquoise']
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, classifier in enumerate(classifiers):
        performance = [results[feature_type][classifier] for feature_type in feature_types]
        bar_position = index + i * bar_width
        plt.bar(bar_position, performance, bar_width, alpha=opacity, color=colors[i], label=classifier)

    plt.xlabel('Feature Types')
    plt.ylabel('Accuracy')
    plt.title('Feature Type Performance by Classifier')
    plt.xticks(index + bar_width * 2.5, feature_types)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the layout to make space for the legend
    plt.ylim(0.5, 1.0)  # Adjust the limits according to your data

    plt.show()

def PCA():
    feature_array, feature_names = self.feature_selection()
    pca = PCA(n_components= feature_array.shape[1] // 5)
    print("Feature Array Shape Before: " + str(feature_array.shape))
    feature_array = pca.fit_transform(feature_array)
    print("Feature Array Shape: " + str(feature_array.shape))
    explained_variance_ratios = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratios)
    components = range(1, len(explained_variance_ratios) + 1)
    plt.bar(components, explained_variance_ratios, align='center', alpha=0.5, label='Individual explained variance')
    plt.plot(components, cumulative_variance_ratio, 'r-o', label='Cumulative explained variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained (%)')
    plt.title('Explained Variance Ratio of PCA Components')
    plt.xticks(components)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def graph_cm(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

def threshold_finder(model, X_test, y_test):
    y_predict_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba)
    auc = roc_auc_score(y_test, y_predict_proba)

    precision, recall, thresholds2 = precision_recall_curve(y_test, y_predict_proba)


    f1_scores = 2 * (precision * recall) / (precision + recall)

    # Plot ROC Curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="red", label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="black", ls="--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('1 - Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.legend(prop={'size':12}, loc=4)

    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(thresholds2, precision[:-1], label="Precision")
    plt.plot(thresholds2, recall[:-1], label="Recall")
    plt.plot(thresholds2, f1_scores[:-1], label="F1-Score")
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend(loc=0)
    plt.xlim([0.025, thresholds2[np.argmin(abs(precision - recall))] + 0.2])
    plt.axvline(thresholds2[np.argmin(abs(precision - recall))], color="k", ls="--")
    plt.title(label=f"Optimal Threshold = {thresholds2[np.argmin(abs(precision - recall))]:.3f}", fontsize=12)
    
    # Display plots
    plt.tight_layout()
    plt.show()

    print(f"AUC: {auc:.3f}")

def plot_two_time_series(series1, series2, labels=('Original Signal', 'Noisy Signal', 'Difference / Added Noise')):

    # Calculate the difference between the signals
    difference = series2 - series1
    
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
    
    # Plot the original signal
    ax1.plot(series1, color='black')
    ax1.set_title(labels[0])
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Amplitude Fp2')
    
    # Plot the noisy signal
    ax2.plot(series2, color='black')
    ax2.set_title(labels[1])
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Amplitude Fp2')
    
    # Plot the difference
    ax3.plot(difference, color='red')
    ax3.set_title(labels[2])
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Difference')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    start = 0
    end = 20 * np.pi  # One full cycle
    step = 0.01

    # Create an array of angles
    angles = np.arange(start, end, step)

    # Compute the sine values
    sine_values = np.sin(angles)
    print(len(sine_values))
    plt.figure(figsize=(10, 6))
    plt.plot(angles, sine_values * 10 ** -7, label='Sine Curve')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Sine value')
    plt.title('Sine Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    x = np.load('npy_arrays/s_data_array.npy')[0][0]
    plot_two_time_series(x, noise_helper(x))
    