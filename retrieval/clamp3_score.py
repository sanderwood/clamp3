import os
import numpy as np
import argparse

def load_npy_files(folder_path):
    """
    Load all .npy files from a specified folder and return a list of numpy arrays.
    """
    npy_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            np_array = np.load(file_path).squeeze()
            npy_list.append(np_array)
    return npy_list

def average_npy(npy_list):
    """
    Compute the average of a list of numpy arrays.
    """
    return np.mean(npy_list, axis=0)

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two numpy arrays.
    """
    dot_product = np.dot(vec1, vec2)
    
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return cosine_sim

if __name__ == '__main__':
    # Set up argument parsing for input folders
    parser = argparse.ArgumentParser(description="Calculate cosine similarity between average feature vectors.")
    parser.add_argument('reference', type=str, help='Path to the reference folder containing .npy files.')
    parser.add_argument('test', type=str, help='Path to the test folder containing .npy files.')
    args = parser.parse_args()

    reference = args.reference
    test = args.test
    # Load .npy files
    ref_npy = load_npy_files(reference)
    test_npy = load_npy_files(test) 
    
    # Compute the average of each list of numpy arrays
    avg_ref = average_npy(ref_npy)
    avg_test = average_npy(test_npy)

    # Compute the cosine similarity between the two averaged numpy arrays
    similarity = cosine_similarity(avg_ref, avg_test)

    # Output the cosine similarity rounded to four decimal places
    print(f"Cosine similarity between '{reference}' and '{test}': {similarity:.4f}")
