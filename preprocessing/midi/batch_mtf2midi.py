input_dir = "<path_to_your_mtf_files>"  # Replace with the path to your folder containing MTF (.mtf) files

import os
import math
import mido
import random
from tqdm import tqdm
from multiprocessing import Pool

def str_to_msg(str_msg):
    type = str_msg.split(" ")[0]
    try:
        msg = mido.Message(type)
    except:
        msg = mido.MetaMessage(type)

    if type in ["text", "copyright", "track_name", "instrument_name", 
                "lyrics", "marker", "cue_marker", "device_name"]:
        values = [type, " ".join(str_msg.split(" ")[1:-1]).encode('utf-8').decode('unicode_escape'), str_msg.split(" ")[-1]]
    elif "[" in str_msg or "(" in str_msg:
        is_bracket = "[" in str_msg
        left_idx = str_msg.index("[") if is_bracket else str_msg.index("(")
        right_idx = str_msg.index("]") if is_bracket else str_msg.index(")")
        list_str = [int(num) for num in str_msg[left_idx+1:right_idx].split(", ")]
        if not is_bracket:
            list_str = tuple(list_str)
        values = str_msg[:left_idx].split(" ") + [list_str] + str_msg[right_idx+1:].split(" ")
        values = [value for value in values if value != ""]
    else:
        values = str_msg.split(" ")

    if len(values) != 1:
        for idx, (key, content) in enumerate(msg.__dict__.items()):
            if key == "type":
                continue
            value = values[idx]
            if isinstance(content, int) or isinstance(content, float):
                float_value = float(value)
                value = float_value
                if value % 1 == 0:
                    value = int(value)
            setattr(msg, key, value)

    return msg

def convert_mtf2midi(file_list):
    for file in tqdm(file_list):
        filename = file.split('/')[-1]
        output_dir = file.split('/')[:-1]
        output_dir[0] = output_dir[0] + '_midi'
        output_dir = '/'.join(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        try:
            with open(file, 'r', encoding='utf-8') as f:
                msg_list = f.read().splitlines()

            # Build a new MIDI file based on the MIDI messages
            new_mid = mido.MidiFile()
            new_mid.ticks_per_beat = int(msg_list[0].split(" ")[1])

            track = mido.MidiTrack()
            new_mid.tracks.append(track)

            for msg in msg_list[1:]:
                if "unknown_meta" in msg:
                    continue
                new_msg = str_to_msg(msg)
                track.append(new_msg)

            output_file_path = os.path.join(output_dir, os.path.basename(file).replace('.mtf', '.mid'))
            new_mid.save(output_file_path)
        except Exception as e:
            with open('logs/mtf2midi_error_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"Error processing {file}: {str(e)}\n")

if __name__ == '__main__':
    file_list = []
    os.makedirs("logs", exist_ok=True)

    # Traverse the specified folder for MTF files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".mtf"):
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
    pool.map(convert_mtf2midi, file_lists)
