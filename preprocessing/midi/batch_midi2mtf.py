import os
import math
import mido
import random
import argparse
from tqdm import tqdm
from multiprocessing import Pool

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Convert MIDI files to MTF format.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the folder containing MIDI (.midi, .mid) files"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the folder where converted MTF files will be saved"
    )
    parser.add_argument(
        "--m3_compatible",
        action="store_true",
        help="Enable M3 compatibility (remove metadata like text, copyright, lyrics, etc.)"
    )
    return parser.parse_args()

def msg_to_str(msg):
    str_msg = ""
    for key, value in msg.dict().items():
        str_msg += " " + str(value)
    return str_msg.strip().encode('unicode_escape').decode('utf-8')

def load_midi(filename, m3_compatible):
    """
    Load a MIDI file and convert it to MTF format.
    """
    mid = mido.MidiFile(filename)
    msg_list = ["ticks_per_beat " + str(mid.ticks_per_beat)]

    # Traverse the MIDI file
    for msg in mid.merged_track:
        if m3_compatible:
            if msg.is_meta:
                if msg.type in ["text", "copyright", "track_name", "instrument_name", 
                                "lyrics", "marker", "cue_marker", "device_name"]:
                    continue
        str_msg = msg_to_str(msg)
        msg_list.append(str_msg)
    
    return "\n".join(msg_list)

def convert_midi2mtf(file_list, input_dir, output_dir, m3_compatible):
    """
    Converts MIDI files to MTF format.
    """
    for file in tqdm(file_list):
        # Construct the output directory by replacing input_dir with output_dir
        relative_path = os.path.relpath(os.path.dirname(file), input_dir)
        output_folder = os.path.join(output_dir, relative_path)
        os.makedirs(output_folder, exist_ok=True)

        try:
            output = load_midi(file, m3_compatible)

            if not output:
                with open('logs/midi2mtf_error_log.txt', 'a', encoding='utf-8') as f:
                    f.write(file + '\n')
                continue
            else:
                output_file_path = os.path.join(output_folder, os.path.splitext(os.path.basename(file))[0] + '.mtf')
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(output)
        except Exception as e:
            with open('logs/midi2mtf_error_log.txt', 'a', encoding='utf-8') as f:
                f.write(file + " " + str(e) + '\n')
            pass

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)  # Ensure absolute path
    output_dir = os.path.abspath(args.output_dir)  # Ensure absolute path
    m3_compatible = args.m3_compatible  # Get M3 compatibility flag

    file_list = []
    os.makedirs("logs", exist_ok=True)

    # Traverse the specified input folder for MIDI files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith((".mid", ".midi")):
                continue
            filename = os.path.join(root, file).replace("\\", "/")
            file_list.append(filename)

    # Prepare for multiprocessing
    file_lists = []
    random.shuffle(file_list)
    for i in range(os.cpu_count()):
        start_idx = int(math.floor(i * len(file_list) / os.cpu_count()))
        end_idx = int(math.floor((i + 1) * len(file_list) / os.cpu_count()))
        file_lists.append(file_list[start_idx:end_idx])

    # Use multiprocessing to speed up conversion
    pool = Pool(processes=os.cpu_count())
    pool.starmap(
        convert_midi2mtf, 
        [(file_list_chunk, input_dir, output_dir, m3_compatible) for file_list_chunk in file_lists]
    )
