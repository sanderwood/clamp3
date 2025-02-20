import os
import sys
from utils import *

def main():
    if len(sys.argv) != 3:
        print('Usage: python clamp3_score.py <test_dir> <ref_dir>')
        sys.exit(1)

    test_dir = os.path.abspath(sys.argv[1])
    ref_dir = os.path.abspath(sys.argv[2])

    # Step 1: Create a temporary directory
    os.makedirs('temp', exist_ok=True)

    # Step 2: Determine modalities automatically
    test_modality = get_modality_from_dir(test_dir)
    ref_modality = get_modality_from_dir(ref_dir)

    if test_modality is None:
        print(f'Error: Could not determine modality for test_dir "{test_dir}"')
        sys.exit(1)

    if ref_modality is None:
        print(f'Error: Could not determine modality for ref_dir "{ref_dir}"')
        sys.exit(1)

    print(f'Detected test_modality: {test_modality}, ref_modality: {ref_modality}')

    # Step 3: Extract features based on detected modality
    modality_functions = {
        'txt': extract_txt_features,
        'img': extract_img_features,
        'xml': extract_abc_features,
        'mid': extract_mid_features,
        'audio': extract_audio_features,
    }

    if os.path.exists(f'inference/{test_modality}_features'):
        print(f'Warning: {test_modality}_features already exists, skipping extraction.')
    else:
        modality_functions[test_modality](test_dir)

    if os.path.exists(f'inference/{ref_modality}_features'):
        print(f'Warning: {ref_modality}_features already exists, skipping extraction.')
    else:
        modality_functions[ref_modality](ref_dir)
    
    # Step 4: Change directory to the inference folder
    change_directory('inference')

    # Step 5: Run the score comparison script
    run_command(f'python clamp3_score.py {test_modality}_features {ref_modality}_features')

    # Step 6: Clean up
    change_directory('..')
    remove_folder('temp')

if __name__ == '__main__':
    main()