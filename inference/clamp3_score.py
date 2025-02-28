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
    
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".npy"):
                key = '/'.join(root.split('/')[3:]) + '/' + file.split(".")[0]
                file_path = os.path.join(root, file)
                feat = np.load(file_path).squeeze()
                if key not in features:
                    features[key] = [feat]
                    filenames[key] = [file_path]
                else:
                    features[key].append(feat)
                    filenames[key].append(file_path)

    return features, filenames

def calculate_pairwise_similarity(query_features, query_filenames, reference_features, reference_filenames):
    """
    Calculate pairwise similarity between query and reference features.
    Ensure strict matching: if any query or reference file cannot be matched, an error is raised.
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
        print(f"There are {len(unmatched_files)} unmatched query or reference files. Check 'inference/unmatched_files.jsonl' for details.")
        option = input("Do you want to continue with the matched files? (y/n): ")
        if option.lower() != 'y':
            raise ValueError("Exiting due to unmatched query or reference files.")
    
    print(f"There are {len(common_keys)} common keys between query and reference features.")
    
    total_query, total_similarity = 0, 0
    pairwise_json = []
    
    for key in tqdm(common_keys):
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
    
    # Sort the pairwise similarity scores by similarity value
    pairwise_json = sorted(pairwise_json, key=lambda x: x['similarity'], reverse=True)
    
    # Save the pairwise similarity scores in a JSONL file
    with open('pairwise_similarities.jsonl', 'w') as f:
        for item in pairwise_json:
            f.write(json.dumps(item) + '\n')
    
    print(f"Total query features: {total_query}")
    print(f"Total reference features: {len(reference_features)}")
    print(f"Avg. pairwise similarity: {round(total_similarity / total_query, 4)}")
    print("Pairwise similarity scores saved in 'inference/pairwise_similarities.jsonl'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate similarity metrics between query and reference features.")
    parser.add_argument('query_folder', type=str, help='Path to the folder containing query features (.npy files).')
    parser.add_argument('reference_folder', type=str, help='Path to the folder containing reference features (.npy files).')
    parser.add_argument('--group', action='store_true', help='Calculate group similarity instead of pairwise similarity.')
    args = parser.parse_args()

    print("Query folder:", args.query_folder)
    print("Reference folder:", args.reference_folder)
    
    query_features, query_filenames = get_features(args.query_folder)
    reference_features, reference_filenames = get_features(args.reference_folder)

    # Calculate similarity metrics
    if not args.group:
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