import os
import sys
from utils import *

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('Usage: python clamp3_score.py <query_dir_path> <ref_dir_path> [--group]')
        sys.exit(1)

    query_dir_path = os.path.abspath(sys.argv[1])
    ref_dir_path = os.path.abspath(sys.argv[2])

    query_dir = os.path.basename(query_dir_path)
    ref_dir = os.path.basename(ref_dir_path)

    group_flag = '--group' if len(sys.argv) == 4 and sys.argv[3] == '--group' else None

    # Step 1: Create temporary and cache directories
    os.makedirs('temp', exist_ok=True)
    os.makedirs('cache', exist_ok=True)

    # Step 2: Determine modalities automatically
    query_modality = get_modality_from_dir(query_dir_path)
    ref_modality = get_modality_from_dir(ref_dir_path)

    if query_modality is None:
        print(f'Error: Could not determine query modality for "{query_dir}"')
        sys.exit(1)

    if ref_modality is None:
        print(f'Error: Could not determine reference modality for "{ref_dir}"')
        sys.exit(1)

    print(f'Detected query modality: {query_modality}')
    print(f'Detected reference modality: {ref_modality}')

    # Step 3: Extract features based on detected modality
    modality_functions = {
        'txt': extract_txt_features,
        'img': extract_img_features,
        'xml': extract_xml_features,
        'mid': extract_mid_features,
        'audio': extract_audio_features,
    }

    if os.path.exists(f'cache/{query_modality}-{query_dir}'):
        print(f'Warning: {query_dir} already exists in the cache, skipping extraction.')
    else:
        modality_functions[query_modality](query_dir_path, f'cache/{query_modality}-{query_dir}')

    if os.path.exists(f'cache/{ref_modality}-{ref_dir}'):
        print(f'Warning: {ref_dir} already exists in the cache, skipping extraction.')
    else:
        modality_functions[ref_modality](ref_dir_path, f'cache/{ref_modality}-{ref_dir}')
    
    # Step 4: Change directory to the inference folder
    change_directory('inference')

    # Step 5: Run the score comparison script with or without --group
    if group_flag:
        run_command(f'python clamp3_score.py ../cache/{query_modality}-{query_dir} ../cache/{ref_modality}-{ref_dir} --group')
    else:
        run_command(f'python clamp3_score.py ../cache/{query_modality}-{query_dir} ../cache/{ref_modality}-{ref_dir}')

    # Step 6: Clean up
    change_directory('..')
    remove_folder('temp')

if __name__ == '__main__':
    main()
