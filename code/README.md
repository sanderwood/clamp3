# CLaMP 3 Codebase

## Overview
CLaMP 3 is a system designed to enhance music information retrieval (MIR) across various musical modalities and languages. Using contrastive learning, it aligns sheet music, audio, and multilingual text into a shared representation space, achieving top performance on MIR tasks. Below is a description of the scripts in the `code/` folder.

## Repository Structure
The `code/` folder contains the following scripts:

### 1. `config.py`
This script contains the training hyperparameters and file paths for the `train_clamp3.py` and `train_m3.py` scripts. You can adjust parameters like learning rates, batch sizes, and data file locations for training.

By default, the configuration is set to the **SAAS version** of CLaMP 3, which provides optimal performance, especially for audio data.

For better performance with symbolic music data, the **C2 variant** of CLaMP 3 is recommended. However, note that its performance on audio data is less optimal. To switch to the C2 variant, modify line 66 of `config.py`, changing the setting from **SAAS** to **C2**.

#### Download Weights
- [CLaMP 3 SAAS (Optimal for Audio)](https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth)
- [CLaMP 3 C2 (Optimal for Symbolic)](https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_c2_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth)
- [M3 Model (Pre-trained Symbolic Music Encoder)](https://huggingface.co/sander-wood/clamp2/blob/main/weights_m3_p_size_64_p_length_512_t_layers_3_p_layers_12_h_size_768_lr_0.0001_batch_16_mask_0.45.pth)

> **Note**: M3 refers to CLaMP 3's symbolic music encoder. If you don't need to retrain or fine-tune M3, you can skip modifying its related settings.

### 2. `extract_clamp3.py`
This script uses the pre-trained CLaMP 3 model to extract representations from text (.txt), sheet music (.abc), MIDI (.mtf), or pre-extracted audio features (.npy). The input files are stored in a specified folder, and the extracted features are saved to a target output folder in `.npy` format.

For data preprocessing, use the scripts in the `preprocessing/` folder:
- **Text files** (.txt) are processed directly.
- **Sheet music** (.abc) is preprocessed into interleaved ABC notation using scripts in `preprocessing/abc/`.
- **MIDI** files (.mtf) are preprocessed into MTF format using `preprocessing/midi/midi2mtf.py`.
- **Audio features** (.npy), extracted using MERT, are processed using `preprocessing/audio/extract_mert.py`.

Regardless of the input type, all extracted features are saved as `.npy` files. These features can either be transformed into global semantic vectors (via average pooling and a linear layer) or retain temporal information by extracting the hidden states from the last layer. 

> **Note**: In this project, we use the global semantic vectors for both classification and retrieval tasks.

**Usage**:

```bash
accelerate launch extract_clamp3.py --epoch <epoch> <input_dir> <output_dir> [--get_global]
```

- `--epoch <epoch>`: (Optional) The epoch of the checkpoint to load.
- `input_dir`: Directory containing input data files.
- `output_dir`: Directory to save the output features.
- `--get_global`: (Optional) Flag to extract global features (semantic vectors). If not specified, the script will extract features with temporal information.

This script supports multi-GPU processing using `accelerate launch` for efficient extraction across multiple GPUs.

### 3. `extract_m3.py`
This script uses the pre-trained **M3** model, CLaMP 3's symbolic music encoder, to extract representations from sheet music in interleaved ABC notation and MTF format. The extracted features are saved as `.npy` files.

Unlike `extract_clamp3.py`, which aligns text and music for global semantic vector extraction, this script does not align with text and retains only temporal information. Each feature corresponds to a patch (a bar in ABC notation or a MIDI message).

**Usage**:

```bash
accelerate launch extract_m3.py <input_dir> <output_dir>
```

- `input_dir`: Directory containing input files (in .abc or .mtf format).
- `output_dir`: Directory to save the extracted features.

### 4. `train_clamp3.py`
This script manages the training process for CLaMP 3. It can be used with two separate training scripts depending on the modality:

- `train_clamp3_symbolic.py` for symbolic music data (.abc, .mtf).
- `train_clamp3_audio.py` for MERT-extracted audio features (.npy).

#### CLaMP 3 Core Components:
- **Text Encoder**: Based on XLM-R-base for cross-lingual text processing.
- **Symbolic Music Encoder (M3)**: Encodes ABC and MIDI, treating each segment (bar or MIDI message) as a patch.
- **Audio Encoder**: A transformer trained on MERT-extracted features, processing up to 640 seconds of audio.

Each encoder produces a global semantic feature via average pooling.

Training data paths are defined in `config.py` under `TRAIN_JSONL`, and validation uses `EVAL_JSONL` by default.

**Training Command**:
For symbolic music:
```bash
python -m torch.distributed.launch --nproc_per_node=<number_of_GPUs> --use_env train_clamp3_symbolic.py
```
For audio:
```bash
python -m torch.distributed.launch --nproc_per_node=<number_of_GPUs> --use_env train_clamp3_audio.py
```

Replace `<number_of_GPUs>` with the number of GPUs for training.

**Input Data Format**:
The input training data should be in **JSONL** format, with each line containing a single JSON object structured as follows:

```json
{
  "filepaths": ["List of file paths to music files"],
  "id": "Unique identifier for the music entry",
  "title": "Music title",
  "artists": ["List of artists"],
  "region": "Region associated with the music",
  "language": "Language of the music",
  "genres": ["List of genres"],
  "tags": ["List of tags/keywords"],
  "background": "Background information",
  "analysis": "Detailed musical analysis",
  "description": "Music description",
  "scene": "Scene description",
  "translations": {
    "language": "Translated language",
    "background": "Translated background",
    "analysis": "Translated analysis",
    "description": "Translated description",
    "scene": "Translated scene"
  }
}
```

### 5. `train_m3.py`
This script trains the **M3** model, which encodes interleaved ABC and MTF files. The directories for training and optional evaluation data should be specified in the `TRAIN_FOLDERS` and `EVAL_FOLDERS` variables, respectively.

> **Note**: Typically, retraining the M3 model is unnecessary for most users, especially those focused on CLaMP 3 or working exclusively with audio. The [pre-trained M3 model](https://huggingface.co/sander-wood/clamp2/blob/main/weights_m3_p_size_64_p_length_512_t_layers_3_p_layers_12_h_size_768_lr_0.0001_batch_16_mask_0.45.pth) is generally sufficient.

**Training Command**:

```bash
python -m torch.distributed.launch --nproc_per_node=<number_of_GPUs> --use_env train_m3.py
```

Replace `<number_of_GPUs>` with the number of GPUs you want to use for training.

### 6. `utils.py`
This utility script contains various classes and functions for model definitions and training utilities.
