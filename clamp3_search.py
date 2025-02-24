import os
import sys
import shutil
from utils import *

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print('Usage: python clamp3_search.py <query_file> <ref_dir> [--top_k TOP_K]')
        sys.exit(1)

    query_file_path = os.path.abspath(sys.argv[1])
    ref_dir_path = os.path.abspath(sys.argv[2])

    query_file = os.path.basename(query_file_path)
    ref_dir = os.path.basename(ref_dir_path)

    top_k = 10

    if len(sys.argv) == 5 and sys.argv[3] == '--top_k':
        try:
            top_k = int(sys.argv[4])
        except ValueError:
            print("Error: --top_k should be an integer.")
            sys.exit(1)

    # Step 1: Create temporary and cache directories
    os.makedirs('temp/query', exist_ok=True)
    os.makedirs('cache/query', exist_ok=True)

    # Step 2: Copy the query file to the temporary directory
    temp_query_file = os.path.join('temp/query', query_file)
    shutil.copy2(query_file_path, temp_query_file)

    # Step 3: Determine modalities automatically
    query_modality = get_modality_from_dir('temp/query')
    ref_modality = get_modality_from_dir(ref_dir_path)

    if query_modality is None:
        print(f'Error: Could not determine query modality for "{query_file}"')
        sys.exit(1)

    if ref_modality is None:
        print(f'Error: Could not determine reference modality for "{ref_dir}"')
        sys.exit(1)

    print(f'Detected query modality: {query_modality}')
    print(f'Detected reference modality: {ref_modality}')

    # Step 4: Extract features based on detected modality
    modality_functions = {
        'txt': extract_txt_features,
        'img': extract_img_features,
        'xml': extract_xml_features,
        'mid': extract_mid_features,
        'audio': extract_audio_features,
    }

    if os.path.exists(f'cache/query/{query_modality}-{os.path.splitext(query_file)[0]}.npy'):
        print(f'Warning: {query_file} already exists in the cache, skipping extraction.')
    else:
        modality_functions[query_modality]('temp/query', f'cache/query')
        os.rename(f'cache/query/{os.path.splitext(query_file)[0]}.npy', f'cache/query/{query_modality}-{os.path.splitext(query_file)[0]}.npy')

    if os.path.exists(f'cache/{ref_modality}-{ref_dir}'):
        print(f'Warning: {ref_dir} already exists in the cache, skipping extraction.')
    else:
        modality_functions[ref_modality](ref_dir_path, f'cache/{ref_modality}-{ref_dir}')
    
    # Step 5: Change directory to the inference folder
    change_directory('inference')

    # Step 6: Run semantic search with top_k
    run_command(f'python clamp3_search.py ../cache/query/{query_modality}-{os.path.splitext(query_file)[0]}.npy ../cache/{ref_modality}-{ref_dir} --top_k {top_k}')

    # Step 7: Clean up
    change_directory('..')
    remove_folder('temp')

if __name__ == '__main__':
    main()
