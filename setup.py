from load_data import load_schizo, load_ASD
from concat_p import concatenate_p, get_all_features
from concatination import concatenate
import numpy as np

def setup_data():

    schizo_data_array, schizo_label_array = load_schizo()
    asd_data_array, asd_label_array = load_ASD()

    schizo_data_array = np.array(schizo_data_array)
    schizo_label_array = np.array(schizo_label_array)
    asd_data_array = np.array(asd_data_array)
    asd_label_array = np.array(asd_label_array)

    print("Shapes of schizophrenia data and labels:")
    print("schizo_data_array shape:", schizo_data_array.shape)
    print("schizo_label_array shape:", schizo_label_array.shape)

    print("\nShapes of ASD data and labels:")
    print("asd_data_array shape:", asd_data_array.shape)
    print("asd_label_array shape:", asd_label_array.shape)
    
    np.save('npy_arrays/s_data_array.npy', schizo_data_array)
    np.save('npy_arrays/s_label_array.npy', schizo_label_array)
    np.save('npy_arrays/ASD_data_array.npy', asd_data_array)
    np.save('npy_arrays/ASD_label_array.npy', asd_label_array)
    print("Numpy Arrays Saved")

    return schizo_data_array, schizo_label_array, asd_data_array, asd_label_array

def setup_features(schizo_data_array, asd_data_array):
    schizo_all_feature_names = get_all_features()
    asd_all_feature_names = get_all_features()

    schizo_all_feature_array = concatenate(schizo_data_array, schizo_all_feature_names)
    asd_all_feature_array = concatenate(asd_data_array, asd_all_feature_names)

    schizo_all_feature_array = np.array(schizo_all_feature_array)
    asd_all_feature_array = np.array(asd_all_feature_array)


    print("\nShapes of concatenated features:")
    print("schizo_all_feature_array shape:", schizo_all_feature_array.shape)
    print("asd_all_feature_array shape:", asd_all_feature_array.shape)
    
    np.save('npy_arrays/schizo_all_features_array.npy', schizo_all_feature_array)
    np.save('npy_arrays/schizo_all_features_array.npy', schizo_all_feature_names)
    np.save('npy_arrays/ASD_all_feature_names.npy', asd_all_feature_array)
    np.save('npy_arrays/ASD_all_feature_names.npy', asd_all_feature_names)
    
    return schizo_all_feature_array, asd_all_feature_array

if __name__ == "__main__":
    s_data_array, s_label_array, a_data_array, a_label_array = setup_data()
    s_feature_array, a_feature_array = setup_features(s_data_array, a_data_array)