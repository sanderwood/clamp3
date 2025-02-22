import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import json

def get_features(path):
    """
    Load and return feature data and filenames from .npy files in the given directory.
    Each feature is stored in a dictionary with the filename (without extension) as the key.
    """
    features = {}
    filenames = {}
    
    # Traverse all files in the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".npy"):
                key = '/'.join(root.split('/')[1:]) + '/' + file.split(".")[0]
                file = os.path.join(root, file)
                feat = np.load(file).squeeze()
                if key not in features:
                    features[key] = [feat]
                    filenames[key] = [file]
                else:
                    features[key].append(feat)
                    filenames[key].append(file)
    
    return features, filenames

def calculate_pairwise_similarity(query_features, query_filenames, reference_features, reference_filenames):
    """
    Calculate pairwise similarity between query and reference features.
    """
    common_keys = set(query_features.keys()) & set(reference_features.keys())
    print(len(common_keys), "common keys found between query and reference features.")
    total_similarity, total_query = 0, 0
    pairwise_json = []

    for idx, key in enumerate(tqdm(common_keys)):
        # Get all query features for the current key
        query_feats = query_features[key]
        total_query += len(query_feats)

        for i in range(len(query_feats)):
            # Convert query feature to tensor and add batch dimension
            query_feat = torch.tensor(query_feats[i])
            
            # Convert reference feature to tensor
            ref_feat = torch.tensor(reference_features[key])

            # Compute cosine similarity between the query and reference feature
            similarity = torch.nn.functional.cosine_similarity(query_feat, ref_feat, dim=0)

            # Update total similarity
            total_similarity += similarity.item()

            # Save the similarity score
            pairwise_json.append({
                "query": query_filenames[key][i],
                "reference": reference_filenames[key],
                "similarity": similarity.item()
            })
    
    # Save the pairwise similarity scores in a JSONL file
    with open('pairwise_similarities.jsonl', 'w') as f:
        for item in pairwise_json:
            f.write(json.dumps(item) + '\n')
        
    # Compute the final metrics
    print(f"Total query features: {total_query}")
    print(f"Total reference features: {len(reference_features)}")
    print(f"Avg. pairwise similarity: {round(total_similarity / total_query, 4)}")

if __name__ == '__main__':
    # Set up argument parsing for input directories
    parser = argparse.ArgumentParser(description="Calculate similarity metrics between query and reference features.")
    parser.add_argument('query_folder', type=str, help='Path to the folder containing query features (.npy files).')
    parser.add_argument('reference_folder', type=str, help='Path to the folder containing reference features (.npy files).')
    parser.add_argument('--pairwise', action='store_true', help='Calculate pairwise similarity instead of group similarity.')
    args = parser.parse_args()

    print("Query folder:", args.query_folder)
    print("Reference folder:", args.reference_folder)
    
    # Load features from the specified folders
    query_features, query_filenames = get_features(args.query_folder)
    reference_features, reference_filenames = get_features(args.reference_folder)

    # Calculate similarity metrics
    if args.pairwise:
        # Convert reference features and filenames to single values
        reference_features = {k: v[0] for k, v in reference_features.items()}
        reference_filenames = {k: v[0] for k, v in reference_filenames.items()}

        # Calculate pairwise similarity between query and reference features
        calculate_pairwise_similarity(query_features, query_filenames, reference_features, reference_filenames)
    else:
        # Flatten the features into a single list
        query_features = [feat for feats in query_features.values() for feat in feats]
        reference_features = [feat for feats in reference_features.values() for feat in feats]

        # Compute the average of query and reference features
        avg_query = np.mean(query_features, axis=0)
        avg_ref = np.mean(reference_features, axis=0)
        
        # Compute cosine similarity between the average query and reference features
        similarity = torch.nn.functional.cosine_similarity(torch.tensor(avg_query), torch.tensor(avg_ref), dim=0)
        print(f"Total query features: {len(query_features)}")
        print(f"Total reference features: {len(reference_features)}")
        print(f"Group similarity: {round(similarity.item(), 4)}")