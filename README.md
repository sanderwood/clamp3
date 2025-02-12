# **CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages**
[![Homepage](https://img.shields.io/badge/CLaMP%203%20Homepage-Coming%20Soon-lightgrey?style=for-the-badge&logo=home-assistant)](#)
[![Paper](https://img.shields.io/badge/CLaMP%203%20Paper-Coming%20Soon-lightgrey?style=for-the-badge&logo=arxiv)](#)
[![GitHub](https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/sanderwood/clamp3)
[![Demo](https://img.shields.io/badge/CLaMP%203%20Demo-Coming%20Soon-lightgrey?style=for-the-badge&logo=gradio)](#)
[![Hugging Face](https://img.shields.io/badge/Model%20Weights-Hugging%20Face-ffcc00?style=for-the-badge&logo=huggingface)](https://huggingface.co/sander-wood/clamp3/tree/main)
[![Dataset](https://img.shields.io/badge/M4--RAG%20Pretraining%20Dataset-Hugging%20Face-ffcc00?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/sander-wood/m4-rag)
[![Benchmark](https://img.shields.io/badge/WikiMT--X%20Evaluation%20Benchmark-Hugging%20Face-ffcc00?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/sander-wood/wikimt-x)

<p align="center">
  <img src="overview.png" alt="CLaMP 3 Overview" width="50%">
</p>

## **Overview**
CLaMP 3 is a multimodal and multilingual framework for music information retrieval (MIR) that supports all major music formats—sheet music, audio, and performance signals—along with multilingual text. It is trained on 27 languages and can generalize to support 100 languages. Using contrastive learning, CLaMP 3 aligns these different formats into a shared representation space, making cross-modal retrieval seamless. Experiments show that it significantly outperforms previous strong baselines, setting a new state-of-the-art in multimodal and multilingual MIR.

### **Key Features**  
- **Multimodal Support:**  
   - **Sheet Music:** Uses **Interleaved ABC notation**, with a context size of **512 bars**.  
   - **Performance Signals:** Processes **MIDI Text Format (MTF)** data, with a context size of **512 MIDI messages**.  
   - **Audio Recordings:** Works with features extracted by **[MERT](https://arxiv.org/abs/2306.00107)**, with a context size of **640 seconds of audio**.

- **Multilingual Capabilities:**  
   - Trained on **27 languages** and generalizes to all **100 languages** supported by **[XLM-R](https://arxiv.org/abs/1911.02116)**.  

- **Datasets & Benchmarking:**  
   - **[M4-RAG](https://huggingface.co/datasets/sander-wood/m4-rag):** A **large-scale** dataset of **2.31M high-quality music-text pairs** across 27 languages and 194 countries.  
   - **[WikiMT-X](https://huggingface.co/datasets/sander-wood/wikimt-x):** A MIR benchmark containing **1,000 triplets** of sheet music, audio, and diverse text annotations.  

### **Applications**  
CLaMP 3 supports a **wide range of music research tasks**, including but not limited to:  
- **Semantic Retrieval:** Find music based on **descriptions** or retrieve textual metadata for **audio or symbolic** inputs.  
- **Zero-Shot Classification:** Categorize **music by genre, region, or other attributes** without labeled training data.  
- **Music Quality Assessment:** Compute the **semantic distance** between reference and generated music features, similar to **Fréchet Inception Distance (FID)**.  
- **Cross-Modal Generative Model Evaluation:** Assess **text-to-music generation, music captioning**, and **symbolic-to-audio synthesis** models.  
- **Computational Musicology:** By visualizing the distribution of data within the **shared representation space**, researchers can explore regional music patterns, stylistic similarities, and cross-cultural influences. 

Importantly, these applications are **not restricted to any specific music modality or language**, making CLaMP 3 a powerful tool for **diverse music AI research**.

## **Repository Structure**
- **[code/](https://github.com/sanderwood/clamp3/tree/main/code)** → Training & feature extraction scripts.
- **[classification/](https://github.com/sanderwood/clamp3/tree/main/classification)** → Linear classification training and prediction.  
- **[preprocessing/](https://github.com/sanderwood/clamp3/tree/main/preprocessing)** → Convert data into **Interleaved ABC, MTF, or MERT-extracted features**.  
- **[retrieval/](https://github.com/sanderwood/clamp3/tree/main/retrieval)** → Semantic search, retrieval evaluation, and similarity calculations.  

> **Note:** Ensure the model weights are placed in the `code/` folder, and verify the **configuration hyperparameters** before use.

## **Getting Started**
### **Environment Setup**
To set up the environment for CLaMP 3, run:  
```bash
conda env create -f environment.yml
conda activate clamp3
```

### **Data Preparation**
#### **1. Convert Music Data to Compatible Formats**
Before using CLaMP 3, preprocess **MusicXML files** into **Interleaved ABC**, **MIDI files** into **MTF**, and **audio files** into **MERT-extracted features**.

> **Note:** Each script requires a manual edit of the `input_dir` variable at the top of the file before running, **except for the MERT extraction script (`extract_mert.py`), which takes command-line arguments for input and output paths.**

##### **1.1 Convert MusicXML to Interleaved ABC Notation**

CLaMP 3 requires **Interleaved ABC notation** for sheet music. To achieve this, first, convert **MusicXML** (`.mxl`, `.xml`, `.musicxml`) to **standard ABC** using [`batch_xml2abc.py`](https://github.com/sanderwood/clamp3/blob/main/preprocessing/abc/batch_xml2abc.py):

```bash
python batch_xml2abc.py
```
- **Input:** `.mxl`, `.xml`, `.musicxml`  
- **Output:** `.abc` (Standard ABC)
 
Next, process the standard ABC files into **Interleaved ABC notation** using [`batch_interleaved_abc.py`](https://github.com/sanderwood/clamp3/blob/main/preprocessing/abc/batch_interleaved_abc.py):

```bash
python batch_interleaved_abc.py
```
- **Input:** `.abc` (Standard ABC)  
- **Output:** `.abc` *(Interleaved ABC for CLaMP 3)*  

##### **1.2 Convert MIDI to MTF Format**
CLaMP 3 processes **performance signals** in **MIDI Text Format (MTF)**. Convert **MIDI files** (`.mid`, `.midi`) into **MTF format** using [`batch_midi2mtf.py`](https://github.com/sanderwood/clamp3/blob/main/preprocessing/midi/batch_midi2mtf.py):

```bash
python batch_midi2mtf.py
```
- **Input:** `.mid`, `.midi`  
- **Output:** `.mtf` *(MTF for CLaMP 3)*  

##### **1.3 Extract Audio Features using MERT**
For audio processing, CLaMP 3 uses **MERT-extracted features** instead of raw waveforms. Extract **MERT-based features** from raw audio (`.mp3`, `.wav`) using [`extract_mert.py`](https://github.com/sanderwood/clamp3/blob/main/preprocessing/audio/extract_mert.py):

```bash
python extract_mert.py --input_path <input_path> --output_path <output_path> --model_path musichubert_hf/MERT-v1-95M --mean_features
```
- **Input:** `.mp3`, `.wav`  
- **Output:** `.npy` *(Processed audio features for CLaMP 3)*  

### **Training and Feature Extraction**
#### **1. Training Models**
Modify **[config.py](https://github.com/sanderwood/clamp3/blob/main/code/config.py)** to adjust **hyperparameters and data paths**.

To train CLaMP 3 on **symbolic music**, use **[train_clamp3_symbolic.py](https://github.com/sanderwood/clamp3/blob/main/code/train_clamp3_symbolic.py)**:

```bash
python -m torch.distributed.launch --nproc_per_node=<GPUs> --use_env train_clamp3_symbolic.py
```

For **audio data**, use **[train_clamp3_audio.py](https://github.com/sanderwood/clamp3/blob/main/code/train_clamp3_audio.py)**:

```bash
python -m torch.distributed.launch --nproc_per_node=<GPUs> --use_env train_clamp3_audio.py
```

Alternatively, you can use **pre-trained weights**:
- **[CLaMP 3 SAAS (Optimal for Audio)](https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_saas.pth)**
- **[CLaMP 3 C2 (Optimal for Symbolic Music)](https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_c2.pth)**  

By default, CLaMP 3 is configured for the **SAAS version**, which provides **optimal performance on audio data**. If working primarily with **symbolic music**, download the **C2 variant** and modify **line 66 in `config.py`** from **saas** to **c2**.

#### **2. Feature Extraction**
After training (or using pre-trained weights), extract features using [`extract_clamp3.py`](https://github.com/sanderwood/clamp3/blob/main/code/extract_clamp3.py):

```bash
accelerate launch extract_clamp3.py --epoch <epoch> <input_dir> <output_dir> [--get_global]
```
- **`--epoch <epoch>`:** (Optional) Specify the checkpoint epoch.  
- **`<input_dir>`:** Directory containing the input files.  
- **`<output_dir>`:** Destination folder for the output `.npy` features.  
- **`--get_global`**: **(Required for retrieval!)** Extracts a **global semantic vector** for each input.  

All extracted features are stored as `.npy` files.

> **Note**: For retrieval, `--get_global` must be used. Without it, CLaMP 3 will not work correctly for retrieval tasks. You only omit `--get_global` if you are performing downstream fine-tuning or need raw feature extraction for custom tasks.

### **Retrieval and Classification**
#### **1. Semantic Search**
Retrieve **similar music features** using **[`semantic_search.py`](https://github.com/sanderwood/clamp3/tree/main/retrieval/semantic_search.py)**:  
```bash
python semantic_search.py <query_file> <reference_folder> [--top_k TOP_K]
```
> **Note:** Zero-shot classification is essentially **semantic search**, where the query feature is compared against class prototypes.  

#### **2. Classification**
Train a linear classifier using **[`train_cls.py`](https://github.com/sanderwood/clamp3/tree/main/classification/train_cls.py)**:  
```bash
python train_cls.py --train_folder <path> --eval_folder <path> [--num_epochs <int>] [--learning_rate <float>] [--balanced_training]
```
Run inference with **[`inference_cls.py`](https://github.com/sanderwood/clamp3/tree/main/classification/inference_cls.py)**:  
```bash
python inference_cls.py <weights_path> <feature_folder> <output_file>
```

## **Citation**
*Coming Soon...*  