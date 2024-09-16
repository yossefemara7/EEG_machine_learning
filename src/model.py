import ast
import json
import math
import os
import sys
import time
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import scipy.stats as stats
from scipy.stats import entropy, mode

import mne
from tqdm import tqdm
import joblib
from joblib import dump, load

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    BaggingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import shap

from concatination import *


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def bagging_model(features_array, label_array, model = 'svm', test_size = 0.9, n_estimators = 10):
    X_train, X_test, y_train, y_test = train_test_split(features_array, label_array, test_size=test_size, random_state=42)

    models = {
        'svm': {
        'model': SVC(),
        'params': {
            'C': np.logspace(4, 8, 100),
            'kernel': ['rbf']
            }
        },

        'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [20, 30, 40, 50],
            'p': [1, 2],  # corresponds to Manhattan (1) and Euclidean (2)
            'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'extra_trees': {
        'model': ExtraTreesClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'RandomForest': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        }
    }
    chosen_model = models[model]['model']
    chosen_params = models[model]['params']
    print(f"Chosen Bagging model: {chosen_model}")
    
    grid_search = RandomizedSearchCV(chosen_model, chosen_params, cv = 5, n_iter= 3, n_jobs = -1, verbose = 3)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    bagging_clf = BaggingClassifier(estimator = best_model, n_estimators= n_estimators, random_state = 42)
    bagging_clf.fit(X_train, y_train)
    y_pred = bagging_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test accuracy: {accuracy}")
    predictions = bagging_clf.predict(X_test)
    class_report = classification_report(y_test, predictions, output_dict=True)
    confusion_mat = confusion_matrix(y_test, predictions)

    return class_report, [X_train, X_test, y_train, y_test], confusion_mat, bagging_clf, accuracy

def k_fold(feature_array, label_array, k=100):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    k = 1
    for train_index, test_index in kf.split(feature_array):
        print(f'Cur K: {k}')
        X_train, X_test = feature_array[train_index], feature_array[test_index]
        y_train, y_test = label_array[train_index], label_array[test_index]
        
        accuracy, *_ = voting_model(features_array = [X_train, X_test], label_array = [y_train, y_test], helper = True)
        accuracies.append(accuracy)
        k += 1
    mean_accuracy = np.mean(accuracies)
    print(f'K-Fold Validation score {mean_accuracy}')
    return mean_accuracy

def voting_model(features_array, label_array, test_size=0.2, helper = False, feature_bagging = False):
    # Ensure all classifiers that are used with soft voting support predict_proba
    if helper or feature_bagging:
        X_train, X_test = features_array
        y_train, y_test = label_array
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(features_array, label_array, test_size=test_size, random_state=21)
    from helper_functions import plot_calibration_curve
    models = {
        'svm': SVC(kernel='rbf', C=40000, probability=True),
        'knn': KNeighborsClassifier(metric = 'manhattan'),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=500, learning_rate=0.3, max_depth=9),
        'extra_trees': ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=2),
        # 'random_forest': RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2),
        # 'decision_tree': DecisionTreeClassifier(max_depth=None, min_samples_split=2),
        # 'logistic_regression': LogisticRegression(solver='lbfgs', C=1.0, max_iter=100),
        # 'ada_boost': AdaBoostClassifier(n_estimators=50, learning_rate=1.0),
        # 'naive_bayes': GaussianNB(),
        # 'mlp': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', learning_rate='constant', max_iter=200)
    }
    selected_models = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.5f}")
        selected_models.append((name, model))

    both_accuracies = []
    if selected_models:
        # Hard Voting
        hard_voting_clf = VotingClassifier(estimators=selected_models, voting='hard')
        hard_voting_clf.fit(X_train, y_train)
        hard_y_pred = hard_voting_clf.predict(X_test)
        hard_final_accuracy = accuracy_score(y_test, hard_y_pred)
        both_accuracies.append(hard_final_accuracy)
        print(f"Hard Voting Classifier Accuracy with selected models: {hard_final_accuracy:.5f}")
        
        # Soft Voting
        soft_voting_clf = VotingClassifier(estimators=selected_models, voting='soft')
        soft_voting_clf.fit(X_train, y_train)
        soft_y_pred = soft_voting_clf.predict(X_test)
        soft_final_accuracy = accuracy_score(y_test, soft_y_pred)
        both_accuracies.append(soft_final_accuracy)
        print(f"Soft Voting Classifier Accuracy with selected models: {soft_final_accuracy:.5f}")
        
        # Get probability estimates for each sample
        # hard_proba = hard_voting_clf.predict_proba(X_test)
        soft_proba = soft_voting_clf.predict_proba(X_test)
        # print(f"Probabilities for the first sample with Hard Voting: {hard_proba[0]}")
        print(f"Probabilities for the first sample with Soft Voting: {soft_proba[0]}")
        print(f'Actual Diagnosis {y_test[0]} | Predicted {soft_y_pred[0]}')
        # plot_calibration_curve(y_test, soft_proba[: , 1])
    else:
        print("No models achieved an accuracy higher than 0.0")

    if hard_final_accuracy > soft_final_accuracy:
        print("Better result achieved with Hard Voting Classifier")
        voting_type = 'Hard'
        accuracy = hard_final_accuracy
        report = classification_report(y_test, hard_y_pred, output_dict = True)
        confusion_mat = confusion_matrix(y_test, hard_y_pred)
        classifier = hard_voting_clf

    else:
        print("Better result achieved with Soft Voting Classifier")
        voting_type = 'Soft'
        accuracy = soft_final_accuracy
        report = classification_report(y_test, soft_y_pred, output_dict = True)
        confusion_mat = confusion_matrix(y_test, soft_y_pred)
        classifier = soft_voting_clf


    if feature_bagging:
        return soft_proba
    
    return accuracy, report, confusion_mat, classifier, [X_train, X_test, y_train, y_test], both_accuracies, voting_type

def sole_models(features_array, label_array, test_size):
    X_train, X_test, y_train, y_test = train_test_split(features_array, label_array, test_size=test_size, random_state=21)
    
    models = {
        'svm': SVC(),  # Enable probability predictions for SVC
        'knn': KNeighborsClassifier(),
        'gradient_boosting': GradientBoostingClassifier(),
        'extra_trees': ExtraTreesClassifier(),
        'random_forest': RandomForestClassifier(),
        'decision_tree': DecisionTreeClassifier(),
        'logistic_regression': LogisticRegression(),
        'ada_boost': AdaBoostClassifier(),
        'naive_bayes': GaussianNB(),
        'mlp': MLPClassifier()
    }
    
    accuracy_scores = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores[model_name] = accuracy
    
    return accuracy_scores

def meta_classifier(features_array, label_array, test_size = 0.1):
    def prepare_proba(array):
        output = []
        for i in range(len(array[0])):
            temp = []
            for model in array:
                temp.append(model[i][0])
                temp.append(model[i][1])
            output.append(temp)
        return output
    
    X_train, X_test, y_train, y_test = train_test_split(features_array, label_array, test_size=test_size, random_state=21)
    
    from helper_functions import plot_calibration_curve
    models = {
        'svm': SVC(kernel='rbf', C=40000, probability=True),
        'knn': KNeighborsClassifier(metric = 'manhattan'),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=500, learning_rate=0.3, max_depth=9),
        'extra_trees': ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=2),
        'random_forest': RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2),

    }
    probas_test = []
    probas_train = []
    for name, model in models.items():
        print(f'Training Model: {name}')
        model.fit(X_train, y_train)
        y_proba_testing = model.predict_proba(X_test)
        y_proba_training = model.predict_proba(X_train)
        probas_test.append(y_proba_testing) 
        probas_train.append(y_proba_training) 
    
    probas_test = prepare_proba(probas_test)
    probas_train = prepare_proba(probas_train)
    print(f"Probas Test Shape : {np.array(probas_test).shape}")
    print(f"Probas Train Shape : {np.array(probas_train).shape}")
    meta_classifier = LogisticRegression()
    meta_classifier.fit(probas_train, y_train)
    y_pred_meta = meta_classifier.predict(probas_test)
    accuracy = accuracy_score(y_test, y_pred_meta)
    conf_matrix = confusion_matrix(y_test, y_pred_meta)
    report = classification_report(y_test, y_pred_meta, output_dict = True)
    print(f'Accuracy {accuracy}')

    return accuracy, [X_train, X_test, y_train, y_test], report, conf_matrix, meta_classifier


def sole_models_hp_tuning(features_array, label_array, test_size):
    X_train, X_test, y_train, y_test = train_test_split(features_array, label_array, test_size=test_size)
    
    # Define the models
    models = {
        'svm': SVC(probability=True),
        'knn': KNeighborsClassifier(),
        'extra_trees': ExtraTreesClassifier(),
        'random_forest': RandomForestClassifier(),
        'decision_tree': DecisionTreeClassifier(),
        # 'logistic_regression': LogisticRegression(),
        # 'ada_boost': AdaBoostClassifier(),
        # 'naive_bayes': GaussianNB(),
        # 'mlp': MLPClassifier()
    }
    
    # Define parameter grids for each model
    param_grids = {
        'svm': {'C': np.logspace(-2, 6, 5), 'kernel': ['rbf']},
        'knn': {'n_neighbors': [3, 5, 7, 10]},
        'extra_trees': {'n_estimators': [50, 100, 200]},
        'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'decision_tree': {'max_depth': [None, 10, 20]},
        # 'logistic_regression': {'C': [0.1, 1, 10], 'solver': ['liblinear']},
        # 'ada_boost': {'n_estimators': [50, 100, 200]},
        # 'naive_bayes': {},
        # 'mlp': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh']}
    }
    
    # Initialize dictionaries to store results
    accuracy_scores = {}
    best_params = {}
    
    # Perform grid search for each model
    returned_model = None
    best_accuracy = 0
    returned_model_name = None
    for model_name, model in models.items():
        print(f"Tuning {model_name}...")
        param_grid = param_grids.get(model_name, {})
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, scoring='accuracy', verbose = 3, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Predict and score the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores[model_name] = accuracy
        best_params[model_name] = grid_search.best_params_
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            returned_model = best_model
            returned_model_name, model_name

    print(accuracy_scores)
    print(best_params)

    returned_y_pred = returned_model.predict(X_test)
    accuracy = accuracy_score(y_test, returned_y_pred)
    report = classification_report(y_test, returned_y_pred, output_dict = True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    data_split = X_train, X_test, y_train, y_test
    returned_model_info = [accuracy,conf_matrix ,report , data_split, returned_model, returned_model_name]

    return accuracy_scores, best_params, returned_model_info

def feature_bagging_final(data_array, label_array, test_size, feature_loading = False):
    #80-20 testing training
    from helper_functions import get_random_feature_lists, append_to_json, create_test_array, say, more_probable_class
    from concat_p import concatenate_p
    all_models = {
    'svm': SVC(probability=True),
    'knn': KNeighborsClassifier(),
    'extra_trees': ExtraTreesClassifier(),
    'random_forest': RandomForestClassifier(),
    'decision_tree': DecisionTreeClassifier(),
    }
    X_train, X_test, y_train, y_test = train_test_split(data_array, label_array, test_size=test_size, random_state=21)
    feature_lists = get_random_feature_lists(number = 300)
    
    if feature_loading:

        selected_feature_lists = []
        selected_models = []
        selected_parameters = []
        
        results_dict = {}
        features_loaded = 0
        for cur_feature_list in feature_lists:
            good_acc = False
            # print(f"Loading features {cur_feature_list}")
            cur_feature_array = concatenate_p(X_train, cur_feature_list)
            accuracy_scores, parameter_dict = sole_models_hp_tuning(cur_feature_array, y_train, 0.2)
            best_accuracy = 0
            best_model = None
            best_parameters = None
            for model, accuracy in accuracy_scores.items():
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
            best_parameters = parameter_dict[best_model]
            print(f"Best accuracy {best_accuracy}")
            if best_accuracy > 0.95:
                selected_feature_lists.append(cur_feature_list)
                selected_models.append(best_model)
                selected_parameters.append(best_parameters)
                results_dict[str(cur_feature_list)] = {'Accuracy' : best_accuracy,
                                                  'Model' : best_model,
                                                  'Parameters' : best_parameters}
                good_acc = True

            features_loaded += 1
            say(f'{features_loaded} features loaded, 95% accuracy {good_acc}')
        append_to_json('csv_and_txt_files/feature_bagging.json', results_dict)
    else:
        
        selected_feature_lists = []
        selected_models = []
        selected_parameters = []
        with open('csv_and_txt_files/feature_bagging.json', 'r') as file:
            data = json.load(file)
        # print(data)
        extracted_info = []
        for d in data:
            for features, metrics in d.items():
                extracted_info.append((features, metrics))

        # Print extracted information
        for features, metrics in extracted_info:
            # print(f"Features: {features}")
            # print(f"Metrics: {metrics}")
            cur_acc = metrics["Accuracy"]
            if cur_acc > 0.97:
                selected_feature_lists.append(ast.literal_eval(features))
                selected_models.append(metrics['Model'])
                selected_parameters.append(metrics['Parameters'])
                # print("Appended")

    preds = []
    f_num = 0
    training_preds = []
    total_feat = len(selected_feature_lists)
    for i in range(len(selected_feature_lists)):
        # print(f'Loading {len(selected_feature_lists)} features')
        cur_feature_list = selected_feature_lists[i]
        cur_model = all_models[selected_models[i]]
        cur_parameters = selected_parameters[i]
        print(f'Model {cur_model}, Paramters {cur_parameters} on {cur_feature_list}')
        cur_model.set_params(**cur_parameters)
        
        X_train_feat = concatenate_p(X_train, cur_feature_list)
        X_test_feat = concatenate_p(X_test, cur_feature_list)


        cur_model.fit(X_train_feat, y_train)
        y_pred = cur_model.predict(X_test_feat)
        y_pred_training = cur_model.predict(X_train_feat)
        preds.append(y_pred)
        training_preds.append(y_pred_training)
        f_num += 1
        say(f'{f_num} out of {total_feat} Loaded')

    np.save('preds.npy', preds)
    preds = np.load('preds.npy')

    print(f'Preds shape: {preds.shape}')
    all_predictions = np.array(preds)
    combined_predictions = [list(all_predictions[:, i]) for i in range(all_predictions.shape[1])]
    combined_predictions = np.array(combined_predictions)

    print(f'All Predictions Shape: {all_predictions.shape}')


    #########################################################################

    total_sure = 0
    correct_sure = 0
    total = 0
    not_sure_correct = 0
    not_sure_total = 0

    from helper_functions import percentage_of_most_occuring_element, get_proba, calculate_average_proba
    f_list = ['delta_channels','beta_channels', 'rms', 'signal_energy', 'abs_diff_signal', 'min']
    f_array = concatenate_p(X_train, f_list), concatenate_p(X_test, f_list)
    l_array = y_train, y_test
    probas = voting_model(f_array, l_array, feature_bagging = True)
    np.save('probas.npy', probas)
    probas = np.load('probas.npy')
    print(probas.shape)
    for i, pred_array in enumerate(combined_predictions):
        predicted_diagnosis = stats.mode(pred_array)[0][0]
        actual_diagnosis = y_test[i]
        cur_points = percentage_of_most_occuring_element(pred_array)
        print(pred_array)
        print(f"Current Score : {cur_points}")
        print(f"Actual Diagnosis {y_test[i]}")
        if cur_points > 99:
            if actual_diagnosis == predicted_diagnosis:
                correct_sure += 1
            total_sure += 1

        if cur_points < 99:
            #Find average proba between feature bagging and voting classifier
            voting_proba = probas[i]
            feature_baggigng_proba = get_proba(pred_array)
            predicted_diagnosis = calculate_average_proba(voting_proba, feature_baggigng_proba)
            if predicted_diagnosis == actual_diagnosis: not_sure_correct += 1
            not_sure_total += 1
        total += 1
    
    print(f'Not sure total {not_sure_total}')
    print(f'Sure Accuracy {correct_sure / total_sure}')
    print(f'Not Sure Accuracy {not_sure_correct / not_sure_total}')
    print(f'Total Accuracy {(correct_sure + not_sure_correct) / total}')
    accuracy = (correct_sure + not_sure_correct) / total

    return accuracy

if __name__ == '__main__':
    label_array = np.random.randint(0, 2, size=(700,))
    mult_feature_array = np.random.random((6, 700, 20))
