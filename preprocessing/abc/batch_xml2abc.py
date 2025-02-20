import os
import sys
import math
import random
import subprocess
from tqdm import tqdm
from multiprocessing import Pool
import argparse

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Convert XML files to ABC format")
    parser.add_argument(
        'input_dir',
        type=str,
        help="Path to the folder containing XML (.xml, .mxl, .musicxml) files"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help="Path to the folder where converted ABC files will be saved"
    )
    return parser.parse_args()

def convert_xml2abc(file_list, input_dir, output_root_dir):
    cmd = sys.executable + " utils/xml2abc.py -d 8 -x "
    for file in tqdm(file_list):
        # Construct the output directory by replacing input_dir with output_root_dir
        relative_path = os.path.relpath(os.path.dirname(file), input_dir)
        output_dir = os.path.join(output_root_dir, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        try:
            p = subprocess.Popen(cmd + '"' + file + '"', stdout=subprocess.PIPE, shell=True)
            result = p.communicate()
            output = result[0].decode('utf-8')

            if not output:
                with open("logs/xml2abc_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(file + '\n')
                continue
            else:
                output_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0] + '.abc')
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(output)
        except Exception as e:
            with open("logs/xml2abc_error_log.txt", "a", encoding="utf-8") as f:
                f.write(file + ' ' + str(e) + '\n')
            pass

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)  # Ensure absolute path
    output_root_dir = os.path.abspath(args.output_dir)  # Ensure absolute path

    file_list = []
    os.makedirs("logs", exist_ok=True)

    # Traverse the specified input folder for XML/MXL files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith((".mxl", ".xml", ".musicxml")):
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

    # Create a pool of processes to convert files in parallel
    pool = Pool(processes=os.cpu_count())
    pool.starmap(convert_xml2abc, [(file_list_chunk, input_dir, output_root_dir) for file_list_chunk in file_lists])
