# **CLaMP 3 Codebase**
CLaMP 3 is designed to enhance music information retrieval (MIR) across various musical modalities and languages. By leveraging contrastive learning, it aligns sheet music, audio, and multilingual text into a shared representation space, achieving top performance on MIR tasks. This codebase covers configuration, training, and feature extraction scripts for CLaMP 3.

**Download Pre-trained Weights:**
- [**CLaMP 3 SAAS (Optimal for Audio)**](https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth)
- [**CLaMP 3 C2 (Optimal for Symbolic)**](https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_c2_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth)
- [**M3 Model (Pre-trained Symbolic Music Encoder)**](https://huggingface.co/sander-wood/clamp2/blob/main/weights_m3_p_size_64_p_length_512_t_layers_3_p_layers_12_h_size_768_lr_0.0001_batch_16_mask_0.45.pth)

> **Note:** M3 is the symbolic music encoder. If you don't need to retrain CLaMP 3 or fine-tune M3, you can skip it.

## **Repository Structure**  
The [code/](https://github.com/sanderwood/clamp3/tree/main/code) folder contains the following scripts:

### **1. [config.py](https://github.com/sanderwood/clamp3/blob/main/code/config.py)**
This script holds the training hyperparameters and file paths for the main training scripts:
- **[train_clamp3_audio.py](https://github.com/sanderwood/clamp3/blob/main/code/train_clamp3_audio.py)**
- **[train_clamp3_symbolic.py](https://github.com/sanderwood/clamp3/blob/main/code/train_clamp3_symbolic.py)**
- **[train_m3.py](https://github.com/sanderwood/clamp3/blob/main/code/train_m3.py)**

**Key Points:**
- **Default Configuration:** Set to the **SAAS version**, which is optimal for audio data.
- **Switching Variants:** For better performance with symbolic music, switch to the **C2 variant** by modifying line 66 in [config.py](https://github.com/sanderwood/clamp3/blob/main/code/config.py) (change **saas** to **c2**).

### **2. Training Scripts**

#### **a. [train_clamp3_audio.py](https://github.com/sanderwood/clamp3/blob/main/code/train_clamp3_audio.py) & [train_clamp3_symbolic.py](https://github.com/sanderwood/clamp3/blob/main/code/train_clamp3_symbolic.py)**
These scripts manage the training of CLaMP 3 based on the modality:

- **Audio Training:** Use `train_clamp3_audio.py` for MERT-extracted audio features (.npy).
- **Symbolic Music Training:** Use `train_clamp3_symbolic.py` for symbolic music data (.abc, .mtf).

**Core Components:**
- **Text Encoder:** Based on XLM-R-base for cross-lingual processing (up to 128 tokens).
- **Symbolic Music Encoder (M3):** Processes ABC and MIDI patches (up to 512 ABC bars or MIDI messages).
- **Audio Encoder:** A transformer trained on MERT features (up to 640 seconds of audio).

Each encoder generates a **global semantic feature** via average pooling. Training and evaluation data paths are defined in [config.py](https://github.com/sanderwood/clamp3/blob/main/code/config.py) using `TRAIN_JSONL` and `EVAL_JSONL`.

**Training Commands:**

For **Symbolic Music:**
```bash
python -m torch.distributed.launch --nproc_per_node=<number_of_GPUs> --use_env train_clamp3_symbolic.py
```

For **Audio:**
```bash
python -m torch.distributed.launch --nproc_per_node=<number_of_GPUs> --use_env train_clamp3_audio.py
```

#### **b. [train_m3.py](https://github.com/sanderwood/clamp3/blob/main/code/train_m3.py)**
This script trains the **M3** model, which encodes interleaved ABC and MTF files.

**Key Points:**
- Specify the training and evaluation directories in the `TRAIN_FOLDERS` and `EVAL_FOLDERS` variables.
- **Note:** Retraining M3 is generally unnecessary for most users, as the pre-trained M3 model is typically sufficient.

**Training Command:**
```bash
python -m torch.distributed.launch --nproc_per_node=<number_of_GPUs> --use_env train_m3.py
```

### **3. [extract_clamp3.py](https://github.com/sanderwood/clamp3/blob/main/code/extract_clamp3.py)**
This script uses the pre-trained CLaMP 3 model to extract representations from multiple modalities:
- **Text (.txt)**
- **Sheet Music (.abc)**
- **MIDI (.mtf)**
- **Pre-extracted Audio Features (.npy)**

**Preprocessing Guidelines:**
- **Text Files:** Processed directly.
- **Sheet Music (.abc):** Convert to interleaved ABC notation using scripts in [preprocessing/abc/](https://github.com/sanderwood/clamp3/tree/main/preprocessing/abc).
- **MIDI (.mtf):** Process with [batch_midi2mtf.py](https://github.com/sanderwood/clamp3/tree/main/preprocessing/midi/batch_midi2mtf.py).
- **Audio (.npy):** Features extracted with [extract_mert.py](https://github.com/sanderwood/clamp3/blob/main/preprocessing/audio/extract_mert.py).

**Feature Extraction Options:**
- **Global Semantic Vectors:** Via average pooling and a linear layer for classification/retrieval tasks.
- **Temporal Features:** Retain hidden states from the last layer if needed.

**Usage:**
```bash
accelerate launch extract_clamp3.py --epoch <epoch> <input_dir> <output_dir> [--get_global]
```
- **`--epoch <epoch>`:** (Optional) Specify the checkpoint epoch.
- **`<input_dir>`:** Directory containing the input files.
- **`<output_dir>`:** Destination folder for the output `.npy` features.
- **`--get_global`**: (Optional) Flag to extract global semantic vectors instead of temporal features.

### **4. [extract_m3.py](https://github.com/sanderwood/clamp3/blob/main/code/extract_m3.py)**
This script extracts representations from sheet music and MIDI data using the pre-trained **M3** model.

**Key Points:**
- Processes interleaved ABC notation and MTF formats.
- Saves the extracted features as `.npy` files.
- Retains only temporal information (each feature corresponds to a patch such as a bar or MIDI message).

**Usage:**
```bash
accelerate launch extract_m3.py <input_dir> <output_dir>
```
- **`<input_dir>`:** Directory containing input files (.abc or .mtf).
- **`<output_dir>`:** Destination folder for the extracted features.

### **5. [utils.py](https://github.com/sanderwood/clamp3/blob/main/code/utils.py)**
This utility script includes various classes and functions supporting model definitions and training utilities across the CLaMP 3 codebase.