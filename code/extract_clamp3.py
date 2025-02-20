import os
import torch
import numpy as np
from tqdm import tqdm
from config import *
from utils import *
from samplings import *
from accelerate import Accelerator
from transformers import BertConfig, AutoTokenizer
import argparse
import requests

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Feature extraction for CLaMP3.")
parser.add_argument("--epoch", type=str, default=None, help="Epoch of the checkpoint to load.")
parser.add_argument("input_dir", type=str, help="Directory containing input data files.")
parser.add_argument("output_dir", type=str, help="Directory to save the output features.")
parser.add_argument("--get_global", action="store_true", help="Get global feature.")

args = parser.parse_args()

# Retrieve arguments
epoch = args.epoch
input_dir = args.input_dir
output_dir = args.output_dir
get_global = args.get_global

files = []
for root, dirs, fs in os.walk(input_dir):
    for f in fs:
        if f.endswith(".txt") or f.endswith(".abc") or f.endswith(".mtf") or f.endswith(".npy"):
            files.append(os.path.join(root, f))

print(f"Found {len(files)} files in total")

# Initialize accelerator and device
accelerator = Accelerator()
device = accelerator.device
print("Using device:", device)

# Model and configuration setup
audio_config = BertConfig(vocab_size=1,
                        hidden_size=AUDIO_HIDDEN_SIZE,
                        num_hidden_layers=AUDIO_NUM_LAYERS,
                        num_attention_heads=AUDIO_HIDDEN_SIZE//64,
                        intermediate_size=AUDIO_HIDDEN_SIZE*4,
                        max_position_embeddings=MAX_AUDIO_LENGTH)
symbolic_config = BertConfig(vocab_size=1,
                            hidden_size=M3_HIDDEN_SIZE,
                            num_hidden_layers=PATCH_NUM_LAYERS,
                            num_attention_heads=M3_HIDDEN_SIZE//64,
                            intermediate_size=M3_HIDDEN_SIZE*4,
                            max_position_embeddings=PATCH_LENGTH)
model = CLaMP3Model(audio_config=audio_config,
                    symbolic_config=symbolic_config,
                    text_model_name=TEXT_MODEL_NAME,
                    hidden_size=CLAMP3_HIDDEN_SIZE,
                    load_m3=CLAMP3_LOAD_M3)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
patchilizer = M3Patchilizer()

# print parameter number
print("Total Parameter Number: "+str(sum(p.numel() for p in model.parameters())))

# Load model weights
model.eval()
checkpoint_path = CLAMP3_WEIGHTS_PATH
if epoch is not None:
    checkpoint_path = CLAMP3_WEIGHTS_PATH.replace(".pth", f"_{epoch}.pth")

if not os.path.exists(checkpoint_path):
    print("No CLaMP 3 weights found. Downloading from Hugging Face...")
    checkpoint_url = "https://huggingface.co/sander-wood/clamp3/resolve/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
    checkpoint_path = "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"

    response = requests.get(checkpoint_url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(checkpoint_path, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print("Weights file downloaded successfully.")

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
print(f"Successfully Loaded CLaMP 3 Checkpoint from Epoch {checkpoint['epoch']} with loss {checkpoint['min_eval_loss']}")
model.load_state_dict(checkpoint['model'])

def extract_feature(filename, get_global=get_global):
    if not filename.endswith(".npy"):
        with open(filename, "r", encoding="utf-8") as f:
            item = f.read()

    if filename.endswith(".txt"):
        item = list(set(item.split("\n")))
        item = "\n".join(item)
        item = item.split("\n")
        item = [c for c in item if len(c) > 0]
        item = tokenizer.sep_token.join(item)
        input_data = tokenizer(item, return_tensors="pt")
        input_data = input_data['input_ids'].squeeze(0)
        max_input_length = MAX_TEXT_LENGTH
    elif filename.endswith(".abc") or filename.endswith(".mtf"):
        input_data = patchilizer.encode(item, add_special_patches=True)
        input_data = torch.tensor(input_data)
        max_input_length = PATCH_LENGTH
    elif filename.endswith(".npy"):
        input_data = np.load(filename)
        input_data = torch.tensor(input_data)
        input_data = input_data.reshape(-1, input_data.size(-1))
        zero_vec = torch.zeros((1, input_data.size(-1)))
        input_data = torch.cat((zero_vec, input_data, zero_vec), 0)
        max_input_length = MAX_AUDIO_LENGTH
    else:
        raise ValueError(f"Unsupported file type: {filename}, only support .txt, .abc, .mtf, .npy files")

    segment_list = []
    for i in range(0, len(input_data), max_input_length):
        segment_list.append(input_data[i:i+max_input_length])
    segment_list[-1] = input_data[-max_input_length:]

    last_hidden_states_list = []

    for input_segment in segment_list:
        input_masks = torch.tensor([1]*input_segment.size(0))
        if filename.endswith(".txt"):
            pad_indices = torch.ones(MAX_TEXT_LENGTH - input_segment.size(0)).long() * tokenizer.pad_token_id
        elif filename.endswith(".abc") or filename.endswith(".mtf"):
            pad_indices = torch.ones((PATCH_LENGTH - input_segment.size(0), PATCH_SIZE)).long() * patchilizer.pad_token_id
        else:
            pad_indices = torch.ones((MAX_AUDIO_LENGTH - input_segment.size(0), AUDIO_HIDDEN_SIZE)).float() * 0.
        input_masks = torch.cat((input_masks, torch.zeros(max_input_length - input_segment.size(0))), 0)
        input_segment = torch.cat((input_segment, pad_indices), 0)

        if filename.endswith(".txt"):
            last_hidden_states = model.get_text_features(text_inputs=input_segment.unsqueeze(0).to(device),
                                                         text_masks=input_masks.unsqueeze(0).to(device),
                                                         get_global=get_global)
        elif filename.endswith(".abc") or filename.endswith(".mtf"):
            last_hidden_states = model.get_symbolic_features(symbolic_inputs=input_segment.unsqueeze(0).to(device),
                                                          symbolic_masks=input_masks.unsqueeze(0).to(device),
                                                          get_global=get_global)
        else:
            last_hidden_states = model.get_audio_features(audio_inputs=input_segment.unsqueeze(0).to(device),
                                                          audio_masks=input_masks.unsqueeze(0).to(device),
                                                          get_global=get_global)
        if not get_global:
            last_hidden_states = last_hidden_states[:, :input_masks.sum().long().item(), :]
        last_hidden_states_list.append(last_hidden_states)

    if not get_global:
        last_hidden_states_list = [last_hidden_states[0] for last_hidden_states in last_hidden_states_list]
        last_hidden_states_list[-1] = last_hidden_states_list[-1][-(len(input_data)%max_input_length):]
        last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
    else:
        full_chunk_cnt = len(input_data) // max_input_length
        remain_chunk_len = len(input_data) % max_input_length
        if remain_chunk_len == 0:
            feature_weights = torch.tensor([max_input_length] * full_chunk_cnt, device=device).view(-1, 1)
        else:
            feature_weights = torch.tensor([max_input_length] * full_chunk_cnt + [remain_chunk_len], device=device).view(-1, 1)
        
        last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
        last_hidden_states_list = last_hidden_states_list * feature_weights
        last_hidden_states_list = last_hidden_states_list.sum(dim=0) / feature_weights.sum()

    return last_hidden_states_list

def process_directory(input_dir, output_dir, files):
    # calculate the number of files to process per GPU
    num_files_per_gpu = len(files) // accelerator.num_processes

    # calculate the start and end index for the current GPU
    start_idx = accelerator.process_index * num_files_per_gpu
    end_idx = start_idx + num_files_per_gpu
    if accelerator.process_index == accelerator.num_processes - 1:
        end_idx = len(files)

    files_to_process = files[start_idx:end_idx]

    # process the files
    for file in tqdm(files_to_process):
        output_subdir = output_dir + os.path.dirname(file)[len(input_dir):]
        try:
            os.makedirs(output_subdir, exist_ok=True)
        except Exception as e:
            print(output_subdir + " can not be created\n" + str(e))

        output_file = os.path.join(output_subdir, os.path.splitext(os.path.basename(file))[0] + ".npy")

        if os.path.exists(output_file):
            print(f"Skipping {file}, output already exists")
            continue

        try:
            with torch.no_grad():
                features = extract_feature(file).unsqueeze(0)
            np.save(output_file, features.detach().cpu().numpy())
        except Exception as e:
            print(f"Failed to process {file}: {e}")

# process the files
process_directory(input_dir, output_dir, files)