import os
import numpy as np
import argparse

def load_npy_files(folder_path):
    """
    Load all .npy files from a specified folder and return a list of numpy arrays.
    """
    npy_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.npy'):
                file_path = os.path.join(root, file_name)
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
    parser.add_argument('test', type=str, help='Path to the test folder containing .npy files.')
    parser.add_argument('reference', type=str, help='Path to the reference folder containing .npy files.')
    args = parser.parse_args()

    test = args.test
    reference = args.reference
    # Load .npy files
    test_npy = load_npy_files(test) 
    ref_npy = load_npy_files(reference)
    
    # Compute the average of each list of numpy arrays
    avg_test = average_npy(test_npy)
    avg_ref = average_npy(ref_npy)

    # Compute the cosine similarity between the two averaged numpy arrays
    similarity = cosine_similarity(avg_test, avg_ref)

    # Output the cosine similarity rounded to four decimal places
    print(f"Cosine similarity between '{test}' and '{reference}': {similarity:.4f}")