import os
import sys
import shutil
import subprocess

def get_modality_from_dir(directory):
    '''Recursively determine modality based on file extensions found in the directory.'''
    if not os.path.exists(directory) or not os.path.isdir(directory):
        print(f'Error: Directory "{directory}" does not exist or is not a directory.')
        sys.exit(1)

    extension_mapping = {
        'txt': 'txt',
        'png': 'img', 'jpg': 'img', 'jpeg': 'img',
        'xml': 'xml', 'musicxml': 'xml', 'mxl': 'xml',
        'mid': 'mid', 'midi': 'mid',
        'wav': 'audio', 'mp3': 'audio', 'flac': 'audio', 'ogg': 'audio'
    }

    for root, _, files in os.walk(directory):  # Recursively walk through the directory
        for file in files:
            file_ext = file.lower().split('.')[-1]  # Extract file extension
            if file_ext in extension_mapping:
                return extension_mapping[file_ext]  # Return as soon as we find a match

    return None  # If no file matches any known modality
    
def change_directory(target_dir):
    '''Change the current working directory.'''
    try:
        os.chdir(target_dir)
        print(f'Changed directory to: "{os.getcwd()}"')
    except FileNotFoundError:
        print(f'Error: Directory "{target_dir}" does not exist.')
        sys.exit(1)
    except PermissionError:
        print(f'Error: No permission to access "{target_dir}".')
        sys.exit(1)

def run_command(command):
    '''Execute a command and handle errors.'''
    print(f'Executing: "{command}"')
    try:
        subprocess.run(command, shell=True, check=True)
        print(f'Successfully executed: "{command}"')
    except subprocess.CalledProcessError as e:
        print(f'Error executing command: "{command}"\n{e}')
        sys.exit(1)

def remove_folder(folder):
    '''Cross-platform folder deletion.'''
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
            print(f'Successfully deleted: "{folder}"')
        except Exception as e:
            print(f'Failed to delete "{folder}": {e}')
            sys.exit(1)

def extract_txt_features(txt_dir, feat_dir=None):
    '''Extract text features from XML files.'''
    # Step 0: Convert to absolute path
    txt_dir = os.path.abspath(txt_dir)
    if feat_dir is not None:
        feat_dir = os.path.abspath(feat_dir)
    
    # Step 1: Change directory to the code folder
    change_directory('code')

    # Step 2: Run extract_clamp3.py
    if feat_dir is None:
        run_command(f'python extract_clamp3.py "{txt_dir}" ../cache/txt_features --get_global')
    else:
        feat_dir = os.path.abspath(feat_dir)
        run_command(f'python extract_clamp3.py "{txt_dir}" "{feat_dir}" --get_global')
    
    # Step 3: Change directory back to the main folder
    change_directory('..')
    
def extract_img_features(img_dir, feat_dir=None):
    '''Extract image features from XML files.'''
    # Step 0: Convert to absolute path
    img_dir = os.path.abspath(img_dir)
    if feat_dir is not None:
        feat_dir = os.path.abspath(feat_dir)

    # Step 1: Delete the temp folders
    remove_folder('temp/img_captions')

    # Step 2: Change directory to the code folder
    change_directory('inference')

    # Step 3: Run extract_clamp3.py
    run_command(f'python image_captioning.py "{img_dir}" ../temp/img_captions')

    # Step 4: Change directory back to the main folder
    change_directory('../code')

    # Step 5: Run extract_clamp3.py
    if feat_dir is None:
        run_command(f'python extract_clamp3.py ../temp/img_captions ../cache/img_features --get_global')
    else:
        feat_dir = os.path.abspath(feat_dir)
        run_command(f'python extract_clamp3.py ../temp/img_captions "{feat_dir}" --get_global')
    
    # Step 6: Change directory back to the main folder
    change_directory('..')
    
def extract_xml_features(xml_dir, feat_dir=None):
    '''Extract XML (sheet music) features from XML files.'''
    # Step 0: Convert to absolute path
    xml_dir = os.path.abspath(xml_dir)
    if feat_dir is not None:
        feat_dir = os.path.abspath(feat_dir)

    # Step 1: Delete the temp folders
    remove_folder('temp/std_abc')
    remove_folder('temp/int_abc')
                              
    # Step 2: Change directory to the preprocessing/abc folder
    change_directory('preprocessing/abc')

    # Step 3: Convert XML files to interleaved ABC format
    run_command(f'python batch_xml2abc.py "{xml_dir}" ../../temp/std_abc')
    run_command(f'python batch_interleaved_abc.py ../../temp/std_abc ../../temp/int_abc')

    # Step 4: Change directory back to the code folder
    change_directory('../../code')

    # Step 5: Run extract_clamp3.py
    if feat_dir is None:
        run_command(f'python extract_clamp3.py ../temp/int_abc ../cache/xml_features --get_global')
    else:
        feat_dir = os.path.abspath(feat_dir)
        run_command(f'python extract_clamp3.py ../temp/int_abc "{feat_dir}" --get_global')
    
    # Step 6: Change directory back to the main folder
    change_directory('..')
    
def extract_mid_features(mid_dir, feat_dir=None):
    '''Extract MIDI (performance signal) features from MIDI files.'''
    # Step 0: Convert to absolute path
    mid_dir = os.path.abspath(mid_dir)
    if feat_dir is not None:
        feat_dir = os.path.abspath(feat_dir)

    # Step 1: Delete temp folder
    remove_folder('temp/mtf')

    # Step 2: Change directory to the preprocessing/midi folder
    change_directory('preprocessing/midi')

    # Step 3: Convert MIDI files to MTF format
    run_command(f'python batch_midi2mtf.py "{mid_dir}" ../../temp/mtf --m3_compatible')

    # Step 4: Change directory back to the code folder
    change_directory('../../code')

    # Step 5: Run extract_clamp3.py
    if feat_dir is None:
        run_command(f'python extract_clamp3.py ../temp/mtf ../cache/mid_features --get_global')
    else:
        feat_dir = os.path.abspath(feat_dir)
        run_command(f'python extract_clamp3.py ../temp/mtf "{feat_dir}" --get_global')

    # Step 6: Change directory back to the main folder
    change_directory('..')
    
def extract_audio_features(audio_dir, feat_dir=None):
    '''Extract audio features from audio files.'''
    # Step 0: Convert to absolute path
    audio_dir = os.path.abspath(audio_dir)
    if feat_dir is not None:
        feat_dir = os.path.abspath(feat_dir)
    
    # Step 1: Delete temp folder
    remove_folder('temp/mert')

    # Step 2: Change directory to the preprocessing/audio folder
    change_directory('preprocessing/audio')

    # Step 3: Extract MERT features from audio files
    run_command(f'python extract_mert.py --input_path "{audio_dir}" --output_path ../../temp/mert --model_path m-a-p/MERT-v1-95M --mean_features')

    # Step 4: Change directory back to the code folder
    change_directory('../../code')

    # Step 5: Run extract_clamp3.py
    if feat_dir is None:
        run_command(f'python extract_clamp3.py ../temp/mert ../cache/audio_features --get_global')
    else:
        feat_dir = os.path.abspath(feat_dir)
        run_command(f'python extract_clamp3.py ../temp/mert "{feat_dir}" --get_global')

    # Step 6: Change directory back to the main folder
    change_directory('..')
    