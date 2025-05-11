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
    prefix_num = len(path.split('/'))
    
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".npy"):
                key = '/'.join(root.split('/')[prefix_num:]) + '/' + file.split('.')[0]
                file_path = os.path.join(root, file)
                feat = np.load(file_path).squeeze()
                if key not in features:
                    features[key] = [feat]
                    filenames[key] = [file_path]
                else:
                    features[key].append(feat)
                    filenames[key].append(file_path)

    return features, filenames

def calculate_metrics(query_features, query_filenames, reference_features, reference_filenames):
    """
    Calculate MRR, Hit@1, Hit@10, and Hit@100 metrics based on the similarity 
    between query and reference features.
    """
    common_keys = set(query_features.keys()) & set(reference_features.keys())
    
    unmatched_files = []
    
    # Collect unmatched files
    unmatched_queries = set(query_features.keys()) - common_keys
    unmatched_references = set(reference_features.keys()) - common_keys
    
    # Save unmatched query and reference files in a JSONL file
    for uq in unmatched_queries:
        unmatched_files.append({"unmatched_query": query_filenames[uq]})
    for ur in unmatched_references:
        unmatched_files.append({"unmatched_reference": [reference_filenames[ur]]})
    
    if unmatched_files:
        with open('unmatched_files.jsonl', 'w') as f:
            for item in unmatched_files:
                f.write(json.dumps(item) + '\n')
        raise ValueError(f"There are {len(unmatched_files)} unmatched query or reference files. Check 'inference/unmatched_files.jsonl' for details.")
    
    print(f"There are {len(common_keys)} common keys between query and reference features.")
    
    mrr, hit_1, hit_10, hit_100, total_query = 0, 0, 0, 0, 0
    rank_json = []

    for idx, key in enumerate(tqdm(common_keys)):
        # Get all query features for the current key
        query_feats = query_features[key]
        total_query += len(query_feats)

        for i in range(len(query_feats)):
            # Convert query feature to tensor and add batch dimension
            query_feat = torch.tensor(query_feats[i]).unsqueeze(dim=0)
            
            # Collect all reference features for common keys
            ref_feats = torch.tensor(np.array([reference_features[k] for k in common_keys]))

            # Compute cosine similarity between the query and all reference features
            similarities = torch.cosine_similarity(query_feat, ref_feats)

            # Create a list of (similarity, index) pairs
            indexed_sims = list(enumerate(similarities.tolist()))

            # Sort by similarity in descending order, with idx-based tie-breaking
            sorted_indices = sorted(indexed_sims, key=lambda x: (x[1], x[0] == idx), reverse=True)

            # Extract the sorted rank list
            ranks = [x[0] for x in sorted_indices]

            # Calculate MRR
            mrr += 1 / (ranks.index(idx) + 1)

            # Calculate Hit@1, Hit@10, Hit@100
            if idx in ranks[:100]:
                hit_100 += 1
                if idx in ranks[:10]:
                    hit_10 += 1
                    if idx in ranks[:1]:
                        hit_1 += 1
            
            # Save the rank position
            rank_json.append({
                "query": query_filenames[key][i],
                "reference": reference_filenames[key],
                "rank": ranks.index(idx) + 1
            })
    
    # Sort the rank list by rank position
    rank_json = sorted(rank_json, key=lambda x: x['rank'])
    
    # Save the rank list in a JSONL file
    with open('retrieval_ranks.jsonl', 'w') as f:
        for item in rank_json:
            f.write(json.dumps(item) + '\n')

    # Compute the retrieval metrics
    print(f"Total query features: {total_query}")
    print(f"Total reference features: {len(reference_features)}")
    print(f"MRR: {round(mrr / total_query, 4)}")
    print(f"Hit@1: {round(hit_1 / total_query, 4)}")
    print(f"Hit@10: {round(hit_10 / total_query, 4)}")
    print(f"Hit@100: {round(hit_100 / total_query, 4)}")
    print("Rank details saved in 'inference/retrieval_ranks.jsonl'.")

if __name__ == '__main__':
    # Set up argument parsing for input directories
    parser = argparse.ArgumentParser(description="Calculate retrieval metrics between query and reference features.")
    parser.add_argument('query_folder', type=str, help='Path to the folder containing query features (.npy files).')
    parser.add_argument('reference_folder', type=str, help='Path to the folder containing reference features (.npy files).')
    args = parser.parse_args()

    print("Query folder:", args.query_folder)
    print("Reference folder:", args.reference_folder)
    
    # Load features from the specified folders
    query_features, query_filenames = get_features(args.query_folder)
    reference_features, reference_filenames = get_features(args.reference_folder)

    # Convert reference features and filenames to single values
    reference_features = {k: v[0] for k, v in reference_features.items()}
    reference_filenames = {k: v[0] for k, v in reference_filenames.items()}

    # Calculate and print the metrics
    calculate_metrics(query_features, query_filenames, reference_features, reference_filenames)
