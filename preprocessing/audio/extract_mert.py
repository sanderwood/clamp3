import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from hf_pretrains import HuBERTFeature
from MERT_utils import load_audio, find_audios

target_sr = 24000
is_mono = True
is_normalize = False
crop_to_length_in_sec = None
crop_randomly = False
sliding_window_size_in_sec = 5
sliding_window_overlap_in_percent = 0.0
layer = None
reduction = 'mean'

def mert_infr_features(feature_extractor, audio_file, device):
    try:
        waveform = load_audio(
            audio_file,
            target_sr=target_sr,
            is_mono=is_mono,
            is_normalize=is_normalize,
            crop_to_length_in_sec=crop_to_length_in_sec,
            crop_randomly=crop_randomly,
            device=device,
        )
    except Exception as e:
        print(f"skip audio {audio_file} because of {e}")
    wav = feature_extractor.process_wav(waveform)
    wav = wav.to(device)
    if sliding_window_size_in_sec:
        assert sliding_window_size_in_sec > 0, "sliding_window_size_in_sec must be positive"
        overlap_in_sec = sliding_window_size_in_sec * sliding_window_overlap_in_percent / 100
        wavs = []
        for i in range(0, wav.shape[-1], int(target_sr * (sliding_window_size_in_sec - overlap_in_sec))):
            wavs.append(wav[:, i : i + int(target_sr * sliding_window_size_in_sec)])
        if wavs[-1].shape[-1] < target_sr * 1:
            wavs = wavs[:-1]
        features = []
        for wav_chunk in wavs:
            features.append(feature_extractor(wav_chunk, layer=layer, reduction=reduction))
        features = torch.cat(features, dim=1)
    else:
        features = feature_extractor(wav, layer=layer, reduction=reduction)
    return features

def process_directory(input_path, output_path, files, mean_features=False):
    print(f"Found {len(files)} files in total")
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write("Found " + str(len(files)) + " files in total\n")

    # calculate the number of files to process per GPU
    num_files_per_gpu = len(files) // accelerator.num_processes

    # calculate the start and end index for the current GPU
    start_idx = accelerator.process_index * num_files_per_gpu
    end_idx = start_idx + num_files_per_gpu
    if accelerator.process_index == accelerator.num_processes - 1:
        end_idx = len(files)

    files_to_process = files[start_idx:end_idx]

    for file in tqdm(files_to_process):
        output_dir = os.path.dirname(file).replace(input_path, output_path)
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(output_dir + " can not be created\n" + str(e))
            with open("log.txt", "a") as f:
                f.write(output_dir + " can not be created\n" + str(e) + "\n")

        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0] + ".npy")

        if os.path.exists(output_file):
            print(f"Skipping {file}, output already exists")
            with open("skip.txt", "a", encoding="utf-8") as f:
                f.write(file + "\n")
            continue

        try:
            features = mert_infr_features(feature_extractor, file, device)
            if mean_features:
                features = features.mean(dim=0, keepdim=True)
            features = features.cpu().numpy()
            np.save(output_file, features)
            with open("pass.txt", "a", encoding="utf-8") as f:
                f.write(file + "\n")
        except Exception as e:
            print(f"Failed to process {file}: {e}")
            with open("log.txt", "a", encoding="utf-8") as f:
                f.write("Failed to process " + file + ": " + str(e) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process audio files to extract features.")
    parser.add_argument("--input_path", type=str, required=True, help="Root path of the input audio files.")
    parser.add_argument("--output_path", type=str, required=True, help="Root path to save the output features.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the HuBERT model for feature extraction.")
    parser.add_argument("--mean_features", action="store_true", help="Calculate mean of features along the first dimension.")
    args = parser.parse_args()

    # Set paths from arguments
    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path

    # Collect files to process
    files = []
    for root, dirs, fs in os.walk(input_path):
        for f in fs:
            if f.endswith(("wav", "mp3")):
                files.append(os.path.join(root, f))
    print(f"Found {len(files)} files in total")
    with open("files.json", "w", encoding="utf-8") as f:
        json.dump(files, f)

    # Initialize accelerator and device
    accelerator = Accelerator()
    device = accelerator.device
    print("Using device:", device)

    # Initialize feature extractor with model_path from command-line
    feature_extractor = HuBERTFeature(
        model_path,
        24000,
        force_half=False,
        processor_normalize=True,
    )
    feature_extractor.to(device)
    feature_extractor.eval()

    # Process the files with mean_features flag from command-line
    process_directory(input_path, output_path, files, mean_features=args.mean_features)
