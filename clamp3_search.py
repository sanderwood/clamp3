import os
import sys
import shutil
from utils import *  # Assuming your utility functions are in this file

def main():
    # Manually check arguments
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print('Usage: python semantic_search.py <query_file> <ref_dir> [--top_k TOP_K]')
        sys.exit(1)

    query_file = os.path.abspath(sys.argv[1])
    ref_dir = os.path.abspath(sys.argv[2])

    # Default top_k value
    top_k = 10

    # Check if --top_k argument is provided
    if len(sys.argv) == 5 and sys.argv[3] == '--top_k':
        try:
            top_k = int(sys.argv[4])
        except ValueError:
            print("Error: --top_k should be an integer.")
            sys.exit(1)

    # Step 1: Create a temporary query directory
    temp_query_dir = os.path.abspath('temp/query')
    os.makedirs(temp_query_dir, exist_ok=True)

    # Copy query file to temp/query/
    shutil.copy2(query_file, temp_query_dir)

    # Step 2: Determine modalities automatically
    query_modality = get_modality_from_dir(temp_query_dir)
    ref_modality = get_modality_from_dir(ref_dir)

    if query_modality is None:
        print(f'Error: Could not determine modality for query file "{query_file}"')
        sys.exit(1)

    if ref_modality is None:
        print(f'Error: Could not determine modality for ref_dir "{ref_dir}"')
        sys.exit(1)

    print(f'Detected query_modality: {query_modality}')
    print(f'Detected ref_modality: {ref_modality}')

    # Step 3: Extract features based on detected modality
    modality_functions = {
        'txt': extract_txt_features,
        'img': extract_img_features,
        'xml': extract_xml_features,
        'mid': extract_mid_features,
        'audio': extract_audio_features,
    }

    query_feature_dir = os.path.abspath('inference/query_feature')
    ref_features_dir = os.path.abspath(f'inference/{ref_modality}_features')

    remove_folder(query_feature_dir)
    modality_functions[query_modality](temp_query_dir, query_feature_dir)

    if os.path.exists(ref_features_dir):
        print(f'Warning: {ref_modality}_features already exist, skipping extraction.')
    else:
        modality_functions[ref_modality](ref_dir)
    
    # Step 4: Change directory to the inference folder
    change_directory('inference')

    # Step 5: Move the query feature
    query_file = [f for f in os.listdir(query_feature_dir) if f.endswith(".npy")][0]
    shutil.move(os.path.join(query_feature_dir, query_file), 'query_feature.npy')

    # Step 6: Run semantic search with top_k
    run_command(f'python clamp3_search.py query_feature.npy {ref_modality}_features --top_k {top_k}')

    # Step 7: Clean up
    change_directory('..')
    remove_folder('temp')
    os.remove('inference/query_feature.npy')

if __name__ == '__main__':
    main()
