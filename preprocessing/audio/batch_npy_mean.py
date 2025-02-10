import os
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import argparse

# Function to log failed files
def log_failed(file_path, message, log_file):
    """Log failed files and their error messages"""
    with open(log_file, "a") as log:
        log.write(f"{file_path}: {message}\n")

# Function to process a list of files
def process_files(file_list, base_dir, output_dir, log_file, check_shape=None):
    """Process files by calculating the mean and saving the output"""
    for file in tqdm(file_list, desc="Processing files", unit="file"):
        try:
            tensor = np.load(file)
            if check_shape is None or tensor.shape == tuple(check_shape):
                # Compute mean along the first dimension
                mean_tensor = np.mean(tensor, axis=0)

                # Generate relative output path
                rel_path = os.path.relpath(file, base_dir)
                output_path = os.path.join(output_dir, rel_path)
                output_dir_path = os.path.dirname(output_path)

                # Ensure the output directory exists
                os.makedirs(output_dir_path, exist_ok=True)

                # Save the processed file
                np.save(output_path, mean_tensor)
            else:
                # Log files with unexpected shapes
                log_failed(file, f"Unexpected shape: {tensor.shape}", log_file)
        except Exception as e:
            # Log files that failed to load or process
            log_failed(file, str(e), log_file)

# Main function
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process .npy files to calculate mean tensors.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing .npy files.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for processed files.")
    parser.add_argument("--log_file", type=str, default="log_failed.txt", help="Path to the log file for failed files.")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    log_file = args.log_file

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Gather all .npy files from the input directory
    file_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                file_list.append(file_path)

    # Split file list for multiprocessing
    num_processes = os.cpu_count()
    chunk_size = math.ceil(len(file_list) / num_processes)
    file_chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]

    # Set up multiprocessing pool
    pool = Pool(processes=num_processes)
    
    # Process files in parallel
    pool.starmap(
        process_files, 
        [(chunk, input_dir, output_dir, log_file) for chunk in file_chunks]
    )

    pool.close()
    pool.join()


