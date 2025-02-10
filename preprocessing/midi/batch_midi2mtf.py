input_dir = "<path_to_your_midi_files>"  # Replace with the path to your folder containing MIDI (.midi, .mid) files
m3_compatible = True  # Set to True for M3 compatibility; set to False to retain all MIDI information during conversion.

import os
import math
import mido
import random
from tqdm import tqdm
from multiprocessing import Pool

def msg_to_str(msg):
    str_msg = ""
    for key, value in msg.dict().items():
        str_msg += " " + str(value)
    return str_msg.strip().encode('unicode_escape').decode('utf-8')

def load_midi(filename):
    # Load a MIDI file
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

def convert_midi2mtf(file_list):
    for file in tqdm(file_list):
        filename = file.split('/')[-1]
        output_dir = file.split('/')[:-1]
        output_dir[0] = output_dir[0] + '_mtf'
        output_dir = '/'.join(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        try:
            output = load_midi(file)

            if output == '':
                with open('logs/midi2mtf_error_log.txt', 'a', encoding='utf-8') as f:
                    f.write(file + '\n')
                continue
            else:
                with open(output_dir + "/" + ".".join(filename.split(".")[:-1]) + '.mtf', 'w', encoding='utf-8') as f:
                    f.write(output)
        except Exception as e:
            with open('logs/midi2mtf_error_log.txt', 'a', encoding='utf-8') as f:
                f.write(file + " " + str(e) + '\n')
            pass

if __name__ == '__main__':
    file_list = []
    os.makedirs("logs", exist_ok=True)

    # Traverse the specified folder for MIDI files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".mid") and not file.endswith(".midi"):
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

    pool = Pool(processes=os.cpu_count())
    pool.map(convert_midi2mtf, file_lists)
