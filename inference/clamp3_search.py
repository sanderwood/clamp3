import os
import torch
import numpy as np
import argparse

def get_info(folder_path):
    """
    Load all .npy files from a specified folder and return a dictionary of features.
    """
    files = sorted(os.listdir(folder_path))
    features = {}
    
    for file in files:
        if file.endswith(".npy"):
            key = file.split(".")[0]
            features[key] = np.load(os.path.join(folder_path, file)).squeeze()
    
    return features

def main(query_file, features_folder, top_k=10):
    # Load query feature from the specified file
    query_feature = np.load(query_file)[0]  # Load directly from the query file
    query_tensor = torch.tensor(query_feature).unsqueeze(dim=0)

    # Load key features from the specified folder
    key_features = get_info(features_folder)

    # Prepare tensor for key features
    key_feats_tensor = torch.tensor(np.array([key_features[k] for k in key_features.keys()]))

    # Calculate cosine similarity
    similarities = torch.cosine_similarity(query_tensor, key_feats_tensor)
    ranked_indices = torch.argsort(similarities, descending=True)

    # Get the keys for the features
    keys = list(key_features.keys())
    
    # Set top_k to the minimum of the number of features and the specified top_k
    top_k = min(top_k, len(keys))
    
    # Round to 4 decimal places for similarity scores
    print(f"Top {top_k} results among {len(keys)} candidates:")
    for i in range(top_k):
        print(keys[ranked_indices[i]], round(similarities[ranked_indices[i]].item(), 4))

if __name__ == '__main__':
    # Set up argument parsing for input paths
    parser = argparse.ArgumentParser(description="Find top similar features based on cosine similarity.")
    parser.add_argument('query_file', type=str, help='Path to the query feature file (e.g., ballad.npy).')
    parser.add_argument('features_folder', type=str, help='Path to the folder containing feature files for comparison.')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top similar items to display (default: 10).')
    args = parser.parse_args()

    # Execute the main functionality
    main(args.query_file, args.features_folder, args.top_k)
