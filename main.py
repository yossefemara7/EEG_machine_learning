print("Importing Libraries:")
from load_data import *
print("Load Data Done")
import numpy as np
import concat_p
print("Concatiation Done")
from model import *
from concatination import concatenate
import multiprocessing
import random
print("Libraries Imported Successfully!")
print()

types = ["alcoholic", "schizophrenia", "ASD"]

class DataLoader:
    def __init__(self, data_type, auto = True):
        self.data_type = data_type
        self.data_array, self.label_array = self.load_data(auto = auto)
        print("Loading " + str(data_type))
        print("Data Array Shape: " + str(self.data_array.shape))
        print("Label Array Shape: " + str(self.label_array.shape))

        self.loaded_model_results = []
        self.current_model = None

    def set_data(self, data_array, label_array):
        self.data_array = data_array
        self.label_array = label_array

    def load_data(self, auto = True):
        if self.data_type not in types:
            print("Not a type")
        else:
            if self.data_type == "schizophrenia":
                print()
                if not auto:
                    data_array, label_array = load_schizo()
                else:
                    #Access to directory files, needs to be loaded in setup
                    data_array, label_array = np.load('npy_arrays\\s_data_array.npy'), np.load('npy_arrays\\s_label_array.npy')
                    self.channel_list = ['Fp2', 'F8', 'T4', 'T6', 'O2', 'Fp1', 'F7', 'T3', 'T5', 'O1', 'F4', 'C4', 'P4', 'F3', 'C3', 'P3', 'Fz', 'Cz', 'Pz']
                    print(f'Channel List: " {self.channel_list}')
                    print("Channel List Length  "  + str(len(self.channel_list)))

            if self.data_type == "alcoholic": #Work on Later
                if not auto:
                    train_data_path = "C:\\Users\\96654\\Desktop\\oop_rep\\Alc_Data/*.csv" # The path of the data (make sure to add an extra "/" in the path) then add "/.csv" because the data is stored in csv
                    test_data_path = "C:\\Users\\96654\\Desktop\\oop_rep\\Alc_Data2/*.csv"
                    data_array1, label_array1 = load_alc(train_data_path)
                    data_array2, label_array2 = load_alc(test_data_path)
                    data_array = np.concatenate((data_array1, data_array2))
                    label_array = np.concatenate((label_array1, label_array2))
                else:
                    data_array, label_array = np.load("data.npy"), np.load("label.npy")


            if self.data_type == "ASD":
                if not auto:
                    data_array, label_array = load_ASD()
                else:
                    #Access to directory files, needs to be loaded in setup
                    data_array, label_array = np.load('npy_arrays\\ASD_data_array.npy'), np.load('npy_arrays\\ASD_label_array.npy')
                    self.channel_list = ['Fp1', 'C3', 'P1', 'Oz', 'Fpz', 'F8', 'P8', 'P10', 'PO4', 'O2']
                    print(f'Channel List: " {self.channel_list}')
                    print("Channel List Length  "  + str(len(self.channel_list)))

        return data_array, label_array
    
    def insert_clean(self, inserted_array):
        self.clean_data_array = inserted_array
    
    def extract_features(self, features_list, parallel = False):
        from helper_functions import write_to_file
        save_option = input('Save this? If yes enter name, if no type no: ')
        if self.data_type == "schizophrenia":
            if parallel:
                feature_array = concat_p.concatenate_p(self.data_array, features_list)
            else:
                feature_array = concatenate(self.data_array, features_list)
            print(feature_array.shape)
            if save_option != 'no':
                np.save(save_option, feature_array)
                write_to_file('csv_and_txt_files/feature_array_names.txt', save_option)

        if self.data_type == "alcoholic":
            if parallel:
                feature_array = concat_p.concatenate_p(self.data_array, features_list)
            else:
                feature_array = concatenate(self.data_array, features_list)
            print(feature_array.shape)
        if self.data_type == "ASD":
            if parallel:
                feature_array = concat_p.concatenate_p(self.data_array, features_list)
            else:
                feature_array = concatenate(self.data_array, features_list)
            print(feature_array.shape)
        return feature_array

    def test_random_features(self): #DONE
        from helper_functions import random_features_helper_schizo, random_features_helper_alc, random_features_helper_ASD
        if self.data_type == "schizophrenia":
            num_processes = 14
            stop_event = multiprocessing.Event()
            for i in range(1000):
                processes = []
                for _ in range(num_processes):
                    process = multiprocessing.Process(target=random_features_helper_schizo, args=(self.data_array, self.label_array) )
                    processes.append(process)
                    process.start()
                for process in processes:
                    process.join()

        if self.data_type == "alcoholic":
            num_processes = 3
            stop_event = multiprocessing.Event()
            for i in range(10):
                processes = []
                for _ in range(num_processes):
                    process = multiprocessing.Process(target=random_features_helper_alc, args=(self.data_array, self.label_array) )
                    processes.append(process)
                    process.start()
                for process in processes:
                    process.join()

        if self.data_type == "ASD":
            for i in range(10):
                clean = False
                if int(input("Raw (1) or Clean (2)")) == 1:
                    test_array = self.data_array
                    print("Using raw data")
                else:
                    test_array = self.clean_data_array
                    print("Using cleaned data")
                    clean = True
                num_processes = 16
                stop_event = multiprocessing.Event()
                for i in range(10):
                    processes = []
                    for _ in range(num_processes):
                        process = multiprocessing.Process(target=random_features_helper_ASD, args=(test_array, self.label_array, clean))
                        processes.append(process)
                        process.start()
                    for process in processes:
                        process.join()

    def feature_selection(self, all_feature_array, all_feature_names, cor_limit = 0.8):
        #Loaded arrays from directoryt should be loaded in setup
        from featurewiz import featurewiz
        from statistics import mode
        print("Feature Selection Started")
        feature_array = all_feature_array
        print(f"Feature Array Shape {feature_array.shape}")
        
        
        feature_names = all_feature_names
        all_features = []

        for feature in feature_names:
            for channel in self.channel_list:
                cur_feature = channel + ' ' + feature
                all_features.append(cur_feature)

        df = pd.DataFrame(feature_array, columns=all_features)
        df['diagnosis'] = self.label_array
        output = featurewiz(df, target='diagnosis', corr_limit=cor_limit, verbose=0)
        print(type(output))
        print(len(output[0]))
        print(len(output[0]))

        array1 = np.array(output[0])
        array2 = np.array(output[1])
        print(array1.shape)
        print(array2.shape)
        print(array1)
        remove_array = []
        for data in array2:
            temp = []
            for i, number in enumerate(data):
                if number == 1 or number == 0:
                    remove_array.append(i)
        remove = mode(remove_array)
        print(f"Remove {remove}")
        feature_array = []
        for data in array2:
            temp = []
            for i, number in enumerate(data):
                if i == remove:
                    pass
                else:
                    temp.append(number)
            feature_array.append(temp)
        feature_array = np.array(feature_array)
        print(feature_array.shape)
        return feature_array, array1

    def load_model(self, features_list):        
        from helper_functions import string_to_list, read_npy_arrays
        if self.data_type == "ASD" or self.data_type == "schizophrenia":
            choice = int(input("Bagging (1) | Voting(2) | Feature Bagging (3) | Voting Meta (4) | Individual Models (5) | Compare All (6): "))
            if choice != 3:
                feature_selection_choice = int(input("Automatic Feature Selection (1) or Manual Feature Selection (2) | Saved Feature Selection (3): "))
                if feature_selection_choice == 1:
                    print("Loading Automatic Feature Selection: ")
                    
                    if self.data_type == "schizophrenia":
                        all_feature_array = np.load('schizo_all_features_array.npy')
                        all_feature_names = np.load('schizo_all_feature_names.npy')
                        
                    if self.data_array == "ASD":
                        all_feature_array = np.load('ASD_all_features_array.npy')
                        all_feature_names = np.load('ASD_all_feature_names.npy')

                    feature_array, features_list = self.feature_selection(all_feature_array = all_feature_array,
                                                                          all_feature_names = all_feature_names,
                                                                          cor_limit = 0.8)
                # elif feature_selection_choice == 3:
                #     read_npy_arrays()
                #     feature_array_name = input("Choose which array: ")
                #     feature_array = np.load('npy_arrays/' + feature_array_name + '.npy')

                else:
                    if int(input("Normal (1) | Parallel (2): ")) == 1:
                        parallel_bool = False
                    else:
                        parallel_bool = True
                    print("Loading features: " + str(features_list))
                    feature_array = self.extract_features(features_list, parallel_bool)
            voting_type = None
            test_size = float(input("Enter Test Size: "))
            #DATA BAGGING / Working
            if choice == 1: #DONE
                estimators = int(input('Enter n_estimators: '))
                model_list = ['svm', 'knn', 'gradient_boosting', 'extra_trees', 'RandomForest']
                model_index = int(input(f"Choose Model by index: {model_list}: "))
                start_time = time.time()
                bagging_results = bagging_model(feature_array, self.label_array, model = model_list[model_index], n_estimators = estimators, test_size = test_size)
                report, data_split, confusion_matrix, clf, accuracy = bagging_results
                end_time = time.time()
                print(f"Bagging Model loaded in {end_time - start_time} seconds")
                self.current_model = "Segmented Data Bagging"
            #VOTING ENSEMBLE / Working
            elif choice == 2:
                # print("Shuffling Label Array")
                # self.shuffle_label() 
                start_time = time.time()
                voting_results = voting_model(feature_array, self.label_array, test_size = test_size)
                accuracy, report, confusion_matrix, clf, data_split, _, voting_type = voting_results
                end_time = time.time()
                print(f"Voting Model Loaded in {end_time - start_time} seconds")
                self.current_model = f"{voting_type} Voting"
            #FEATURE BAGGING
            elif choice == 3:
                #Test Size cant change
                if test_size != 0.2:
                    print("Feature Bagging can only be loaded for test size 0.2")
                feature_bagging_results = feature_bagging_final(self.data_array, self.label_array, test_size = 0.2)
                accuracy = feature_bagging_results
                print(f"Feature Bagging Accuracy {accuracy}")
                start_time = time.time()
                end_time = time.time()
                print(f'Feature Bagging Model loded in {end_time - start_time} seconds')
                self.current_model = "Diverse Feature Bagging"

            #META CLASSIFIER / Working
            elif choice == 4: 
                start_time = time.time()
                meta_results = meta_classifier(feature_array, self.label_array, test_size = test_size)
                accuracy, data_split, report, confusion_matrix, clf = meta_results
                end_time = time.time()
                print(f"Meta Classifier Model Loaded in {end_time - start_time} seconds")
                self.current_model = "Meta Classifier"

            #INDIVUAL MODELS
            elif choice == 5:
                print("Evaluation will be done on the best performing model")
                ind_models_accuracy, best_parameters, best_model_info  = sole_models_hp_tuning(feature_array, self.label_array, test_size = test_size)
                accuracy, confusion_matrix, report, data_split, clf, model_name =  best_model_info
                ind_models_accuracy = dict(sorted(ind_models_accuracy.items(), key=lambda item: item[1], reverse=True))
                print()
                for model, accuracy in ind_models_accuracy.items():
                    print(f'{model} accuracy: {accuracy}')
                
                self.current_model = f"Sole Model : {model_name}"
                
            elif choice == 6:
                mode = int(input("Normal Test (1) or Compare Leaning Curves (2): "))
                if mode == 1: #DONE
                    info = {}
                    model_list = ['svm', 'knn', 'gradient_boosting', 'extra_trees', 'RandomForest']
                    bagging_model_index = int(input(f"Choose Model by index: {model_list}: "))
                    n_estimators = int(input("Enter n_estimators (Data Bagging Model): "))
                    all_accuracies = {}
                    #
                    _, report, confusion_matrix, clf, data_split, both_voting_accuracies, _ = voting_model(feature_array, self.label_array, test_size = test_size)
                    info['Hard Voting'] = [data_split, report, confusion_matrix, clf, 'Hard']
                    info['Soft Voting'] = [data_split, report, confusion_matrix, clf, 'Soft']
                    hard_voting, soft_voting = both_voting_accuracies
                    report,data_split ,confusion_matrix ,clf , bagging_accuracy = bagging_model(feature_array, self.label_array,
                                                                model = model_list[bagging_model_index], 
                                                                test_size = test_size, n_estimators = n_estimators)
                    info['Segmented Data Bagging'] = [data_split, report, confusion_matrix, clf, None]
                    meta_accuracy, data_split, report, confusion_matrix, clf = meta_classifier(feature_array, self.label_array, test_size = test_size)
                    info['Meta Classifier'] = [data_split, report, confusion_matrix, clf, None]
                    ind_models_accuracy = sole_models(feature_array, self.label_array, test_size = test_size)                    
                    #                    
                    for model, accuracy in ind_models_accuracy.items():
                        all_accuracies[model] = accuracy
                    all_accuracies['Hard Voting'] = hard_voting
                    all_accuracies['Soft Voting'] = soft_voting
                    all_accuracies['Segmented Data Bagging'] = bagging_accuracy
                    all_accuracies['Meta Classifier'] = meta_accuracy
                    if test_size == 0.2:
                        all_accuracies['Diverse Feature Bagging'] = 0.988523
                    all_accuracies = dict(sorted(all_accuracies.items(), key=lambda item: item[1], reverse=True))
                    print()
                    for model, accuracy in all_accuracies.items():
                        print(f'{model} accuracy: {accuracy}')

                    self.current_model = next(iter(all_accuracies))
                    print(self.current_model)

                    data_split, report, confusion_matrix, clf, voting_type = info[self.current_model]

                elif mode == 2:
                    from helper_functions import graph_learning_curves, save_learning_curve
                    increments = float(input('Enter Increments: '))
                    test_sizes = np.arange(increments, 1.0, increments)
                    test_sizes = test_sizes[:-len(test_sizes)//10]
                    model_list = ['svm', 'knn', 'gradient_boosting', 'extra_trees', 'RandomForest']
                    bagging_model_index = int(input(f"Choose Model by index: {model_list}: "))
                    n_estimators = int(input("Enter n_estimators (Data Bagging Model): "))
                    final_dict = {}
                    for test_size in test_sizes:
                        all_accuracies = {}
                        #Voting, Data bagging, Meta Classifier, individual Models
                        _, _, _, _, _, both_voting_accuracies = voting_model(feature_array, self.label_array, test_size = test_size)
                        hard_voting, soft_voting = both_voting_accuracies
                        
                        _, _, _, _, bagging_accuracy = bagging_model(feature_array, self.label_array, model = model_list[bagging_model_index], test_size = test_size, n_estimators = n_estimators)
                                            
                        meta_accuracy = meta_classifier(feature_array, self.label_array, test_size = test_size)
                        
                        ind_models_accuracy = sole_models(feature_array, self.label_array, test_size = test_size)     
                        for model, accuracy in ind_models_accuracy.items():
                                all_accuracies[model] = accuracy
                        all_accuracies['Hard Voting Model'] = hard_voting
                        all_accuracies['Soft Voting Model'] = soft_voting
                        all_accuracies['Bagging Model'] = bagging_accuracy
                        all_accuracies['Meta Classifier Model'] = meta_accuracy

                        all_accuracies = dict(sorted(all_accuracies.items(), key=lambda item: item[1], reverse=True))
                        final_dict[test_size] = all_accuracies
                        winsound.Beep(1000, 100)

                    print(final_dict)
                    graph_learning_curves(final_dict)
                    
                    name = input("Add name for this dictionary, (n) if you dont want to save")
                    if name != 'n':
                        note = input('Enter Notes')
                        save_learning_curve(final_dict, name, note)

                    self.current_model = "Comparing Learning Curves"

        feature_names = features_list
        if self.current_model != "Diverse Feature Bagging" or "Comparing Learning Curves":
            self.loaded_model_results = [feature_array, data_split, feature_names, report, confusion_matrix, clf, voting_type] 


    def gaussian_noise_evaluation(self, feature_list, show = False):
        from helper_functions import add_gaussian_noise_to_eeg, plot_eeg_data
        # increments = float(input("Enter Increments: "))
        # test_sizes = np.arange(increments, 1.0, increments)
        # print(test_sizes)
        accuracies = []
        X_train, X_test, y_train, y_test = train_test_split(self.data_array, self.label_array, test_size=0.5, random_state=21)
        
        plot_eeg_data(X_test[0])
        X_test = add_gaussian_noise_to_eeg(X_test)
        plot_eeg_data(X_test[0])
        plt.show()

        X_train = concat_p.concatenate_p(X_train, feature_list)
        X_test = concat_p.concatenate_p(X_test, feature_list)
        
        feature_array = X_train, X_test
        label_array = y_train, y_test
        cur_accuracy, *_ = voting_model(feature_array, label_array, helper = True)
        accuracies.append(cur_accuracy)
        print(accuracies)

    def load_best_features(self):
        from helper_functions import get_best_features
        if self.data_type == "schizophrenia" or self.data_type == "ASD":
            self.load_model(get_best_features("features2.txt"))
        if self.data_type == "alcoholic":
            self.load_model(get_best_features("features.txt"))

    def print_best_features(self):
        if self.data_type == "schizophrenia":
            with open("features2.txt", "r") as file:
                lines = file.readlines()
                for line in lines:
                    print(line)

        if self.data_type == "alcoholic":
            with open("features.txt", "r") as file:
                lines = file.readlines()
                for line in lines:
                    print(line)

    def shuffle_label(self):
        random.shuffle(self.label_array)
    ########################################

    def evaluate_model(self):
        if self.data_type == "ASD" or "schizophrenia":
            from helper_functions import print_metrics, graph_cm, graph_learning_curve, graph_PCA, SHAP_ROC_AUC_Threshold, threshold_finder
            #Needed variables: report, feature array, confusion_matrix, data_split
            if self.loaded_model_results is None:
                raise ValueError("No model has been loaded to evaluate")
            feature_array, data_split, feature_names, report, confusion_matrix, model, voting_type = self.loaded_model_results
            print_metrics(report, model_name = self.current_model) 
            print()

            print("-"*50)
            if input("Graph Confusion Matrix (y/n): ") == 'y': #DONE
                graph_cm(confusion_matrix)
            print("-"*50)

            if input("Graph Learning Curve (y/n): ") == 'y':
                increments = float(input("Enter the increments: "))
                graph_learning_curve(feature_array, self.label_array, increments)
            print("-"*50)

            if self.current_model != "Meta Classifier":
                if input("View Threshold Analysis (y/n): ") == 'y':
                    X_train, X_test, y_train, y_test = data_split
                    threshold_finder(model, X_test, y_test)         
                print("-"*50)


            # if input("Test Cross Validation (y/n): ") == 'y':
            #     k_folds = int(input("Enter Folds: "))
            #     print(k_fold_cv_voting_classifier(voting_classifier, feature_array, self.label_array))
            # print("-"*50)


    def get_feature_list(self):
        if self.data_type == 'ASD':
            feature_path = 'csv_and_txt_files\\features5.txt'
        elif self.data_type == 'schizophrenia':
            feature_path = 'csv_and_txt_files\\features2.txt'

        with open(feature_path, 'r') as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        features = [line[:line.find('Accuracy:')].strip() if 'Accuracy:' in line else line for line in lines]
        return features, lines
    
    def graph_individual_models(self):
        test_size = 0.2
        spectral = ['hjorth_complexity', 'hjorth_mobility', 'peak_freq_channels', 'spectral_skewness_channels', 'spectral_mean', 'spectral_variance']
        entropy = ['shannon_entropy', 'spectral_shannon_entropy', 'spectral_wavelet_entropy', 'spectral_permutation_entropy', 'fuzzy_entropy']
        time_domain = ['ptp', 'rms', 'abs_diff_signal', 'min', 'std', 'mean']

        bands_feature_array = np.load('bands_new.npy')
        spectral_feature_array = concat_p.concatenate_p(self.data_array, spectral)
        entropy_feature_array = concat_p.concatenate_p(self.data_array, entropy)
        time_domain_feature_array = concat_p.concatenate_p(self.data_array, time_domain)
        
        accuracies = {'bands' : sole_models(bands_feature_array, self.label_array, test_size),
                      'spectral' : sole_models(spectral_feature_array, self.label_array, test_size),
                      'entropy' : sole_models(entropy_feature_array, self.label_array, test_size),
                      'time_domain': sole_models(time_domain_feature_array, self.label_array, test_size)}
        
        print(accuracies)

if __name__ == "__main__":
    import winsound
    def autism_main():
        ASD_loader = DataLoader("ASD", auto = False) 
        feature_list = ASD_loader.get_feature_list()
        GREEN = '\033[92m'
        RESET = '\033[0m'
        # print()
        for i in range(10):
            print(str(i) + ": " + str(feature_list[0][i]))
        feature_index = int(input(GREEN + "Which Feature do you want to evaluate: " + RESET))
        print("Loading Features: " + feature_list[0][feature_index])
        ASD_loader.load_model(feature_list[0][feature_index])
        ASD_loader.evaluate_model()

    def schizo_main():
        schizo_loader = DataLoader('schizophrenia', auto = False)
        schizo_loader.load_model(['delta_channels','beta_channels', 'rms', 'signal_energy', 'abs_diff_signal', 'min', 'mean'])
        schizo_loader.evaluate_model()
        schizo_loader.gaussian_noise_evaluation(['delta_channels','beta_channels', 'rms', 'signal_energy', 'abs_diff_signal', 'min', 'mean'])
        
    main = int(input("Autism (1) | Schizo (2): "))
    if main == 1:
        autism_main()
    elif main == 2:
        schizo_main()
    else:
        print("Wrong Input")


# CHANGE LOAD MODEL TO LOAD FROM THE INDIVIDUAL CSV


# Testing


#Notes:
#Change epoch length
#Allow for soft and hard voting in voting
#Rewrite meta voting