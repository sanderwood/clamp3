# **CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages**
[![Homepage](https://img.shields.io/badge/CLaMP%203%20Homepage-GitHub-181717?style=for-the-badge&logo=home-assistant)](https://sanderwood.github.io/clamp3/)
[![Paper](https://img.shields.io/badge/CLaMP%203%20Paper-Arxiv-red?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2502.10362)
[![GitHub](https://img.shields.io/badge/CLaMP%203%20Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/sanderwood/clamp3)
[![Demo](https://img.shields.io/badge/CLaMP%203%20Demo-Gradio-green?style=for-the-badge&logo=gradio)](https://huggingface.co/spaces/sander-wood/clamp3)
[![Hugging Face](https://img.shields.io/badge/Model%20Weights-Hugging%20Face-ffcc00?style=for-the-badge&logo=huggingface)](https://huggingface.co/sander-wood/clamp3/tree/main)
[![Dataset](https://img.shields.io/badge/M4--RAG%20Dataset-Hugging%20Face-ffcc00?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/sander-wood/m4-rag)
[![Benchmark](https://img.shields.io/badge/WikiMT--X%20Benchmark-Hugging%20Face-ffcc00?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/sander-wood/wikimt-x)

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
   - **[M4-RAG](https://huggingface.co/datasets/sander-wood/m4-rag):** A web-scale dataset of **2.31M high-quality music-text pairs** across 27 languages and 194 countries.  
   - **[WikiMT-X](https://huggingface.co/datasets/sander-wood/wikimt-x):** A MIR benchmark containing **1,000 triplets** of sheet music, audio, and diverse text annotations.  

### **What Can CLaMP 3 Do?**  

CLaMP 3 unifies diverse music data and text into a shared representation space, enabling the following key capabilities:  

- **Text-to-Music Retrieval**: Finds relevant music based on text descriptions in 100 languages.  
- **Image-to-Music Retrieval**: Matches music that aligns with the scene depicted in the image.  
- **Cross-Modal Music Retrieval**: Enables music retrieval and recommendation across different modalities.  
- **Zero-Shot Music Classification**: Identifies musical attributes such as genres, moods, and styles without labeled training data.  
- **Music Semantic Similarity Evaluation**: Measures semantic similarity between:  
   - **Generated music and its text prompt**, validating how well text-to-music models follow instructions.  
   - **Generated music and reference music**, assessing their semantic similarity, including aspects like style, instrumentation, and musicality.  

For examples demonstrating these capabilities, visit [CLaMP 3 Homepage](https://sanderwood.github.io/clamp3/).

### **Quick Start Guide**  
For users who want to get started quickly without delving into the details, follow these steps:

### **Install Environment**  
```bash
conda create -n clamp3 python=3.10.16 -y
conda activate clamp3
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

### **Overview of `clamp3_*.py` Scripts**  
CLaMP 3 provides the `clamp3_*.py` script series for **single-command data preprocessing, feature extraction, retrieval, and evaluation**. These scripts automatically detect and process different modalities, requiring minimal setup.

**Common Features of `clamp3_*.py` Scripts:**  
- **End-to-End Processing**: Each script handles data preprocessing, feature extraction, and similarity computation in a single run.  
- **Automatic Modality Detection**: Supports **audio (`.mp3`, `.wav`), performance signals (`.mid`, `.midi`), sheet music (`.mxl`, `.musicxml`, `.xml`), images (`.png`, `.jpg`), and text (`.txt`)**.  

- **First-Time Model Download**:  
  - The first time you run any `clamp3_*.py` script, the necessary model weights for **[CLaMP 3 (SAAS)](https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth)** and **[MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M)** will be downloaded automatically.  
  - Once downloaded, the models are cached and will not be re-downloaded in subsequent runs.  

- **Feature Management**:  
  - Extracted features are stored in `inference/` and **are not overwritten** if they already exist. This prevents redundant computations, which is especially useful when processing large datasets since feature extraction can be time-consuming.  
  - **If you need to run retrieval on a different dataset**, manually delete the corresponding modality folder inside `inference/` (e.g., remove `inference/audio_features/` when switching to a new batch of audio files). Otherwise, the script will reuse previously extracted features.  
  - Temporary files are stored in `temp/` and **automatically cleared on each run**.  

- **Same-Modality Requirement**: All files within a folder must belong to the same modality.

#### **[`clamp3_search.py`](https://github.com/sanderwood/clamp3/blob/main/clamp3_search.py) - Running Retrieval Tasks**  
```bash
python clamp3_search.py <query_file> <ref_dir>
```
- **Text-to-Music Retrieval**: Query is a `.txt` file, `ref_dir` contains music files.  
- **Image-to-Music Retrieval**: Query is an image (`.png`, `.jpg`), `ref_dir` contains music files. **BLIP** generates captions for search.  
- **Music-to-Music Retrieval**: Query is a music file, `ref_dir` contains music files (can be same or different modality). Supports **cross-modal retrieval** (e.g., retrieving audio using sheet music).  
- **Zero-Shot Classification**: Query is a music file, `ref_dir` contains **text-based class prototypes** (e.g., `"It is classical"`, `"It is jazz"`). The highest similarity match is the classification result.  

#### **[`clamp3_eval.py`](https://github.com/sanderwood/clamp3/blob/main/clamp3_eval.py) - Evaluating Retrieval Performance**  
```bash
python clamp3_eval.py <query_dir> <ref_dir>
```
- **Measures retrieval accuracy** (MRR, Hit@K) and **semantic similarity** (Avg. pair sim).  
- Requires **matching folder structures and filenames** between `query_dir` and `ref_dir`.  

  **Example folder structure:**  
  ```
  query_dir/  
  ├── en/  
  │   ├── sample1.txt  
  │   ├── sample2.txt  
  │   ├── sample3.txt  
  │   └── ...  
  ├── zh/  
  │   ├── sample1.txt  
  │   ├── sample2.txt  
  │   ├── sample3.txt  
  │   └── ...  
  ```

  ```
  ref_dir/  
  ├── en/  
  │   ├── sample1.wav  
  │   ├── sample2.wav  
  │   ├── sample3.wav  
  │   └── ...  
  ├── zh/  
  │   ├── sample1.wav  
  │   ├── sample2.wav  
  │   ├── sample3.wav  
  │   └── ...  
  ```

  - Here, `query_dir/en/sample1.txt` is paired with `ref_dir/en/sample1.wav`, and `query_dir/zh/sample2.txt` is paired with `ref_dir/zh/sample2.wav`.  
  - The folder structure and filenames must match exactly for correct evaluation.
- **Example Output:**  
  ```
  MRR: 0.2882
  Hit@1: 0.181
  Hit@10: 0.509
  Hit@100: 0.866
  Avg. pair sim: 0.4445
  Total pairs: 1000
  ```
  - **Higher MRR / Hit@K** → Better retrieval performance.  
  - **Higher Avg. pair sim** → Stronger semantic similarity between query-ref pairs.  
  - **Cross-modal similarity is lower due to modality gaps.**  

#### **[`clamp3_score.py`](https://github.com/sanderwood/clamp3/blob/main/clamp3_score.py) - Computing Semantic Similarity**  
```bash
python clamp3_score.py <query_dir> <ref_dir>
```
- **Computes overall semantic similarity** between two datasets.  
- **Differences from [`clamp3_eval.py`](https://github.com/sanderwood/clamp3/blob/main/clamp3_eval.py):**  
  1. **Only returns similarity scores** (no retrieval metrics like MRR/Hit@K).  
  2. **Calculates dataset-level similarity** instead of individual query-ref pairs.  
- Useful for **evaluating generated music vs. ground truth** in generative modeling.  

## **Repository Structure**
- **[code/](https://github.com/sanderwood/clamp3/tree/main/code)** → Training & feature extraction scripts.
- **[classification/](https://github.com/sanderwood/clamp3/tree/main/classification)** → Linear classification training and prediction.  
- **[inference/](https://github.com/sanderwood/clamp3/tree/main/inference)** → Semantic search, retrieval evaluation, and similarity calculations.  
- **[preprocessing/](https://github.com/sanderwood/clamp3/tree/main/preprocessing)** → Convert data into Interleaved ABC, MTF, or MERT-extracted features.  

> **Note:** Ensure the model weights are placed in the `code/` folder, and verify the configuration hyperparameters before use.

## **Key Script Overview**
### **Data Preparation**
#### **1. Convert Music Data to Compatible Formats**
Before using CLaMP 3, preprocess **MusicXML files** into **Interleaved ABC**, **MIDI files** into **MTF**, and **audio files** into **MERT-extracted features**.

##### **1.1 Convert MusicXML to Interleaved ABC Notation**  

CLaMP 3 requires **Interleaved ABC notation** for sheet music. Follow these steps:

1. Convert **MusicXML** (`.mxl`, `.xml`, `.musicxml`) to **standard ABC** using [`batch_xml2abc.py`](https://github.com/sanderwood/clamp3/blob/main/preprocessing/abc/batch_xml2abc.py):  
   ```bash
   python batch_xml2abc.py <input_dir> <output_dir>
   ```
   - **Input:** Directory containing `.mxl`, `.xml`, `.musicxml` files  
   - **Output:** Directory where converted `.abc` (Standard ABC) files will be saved  

2. Convert **Standard ABC** into **Interleaved ABC** using [`batch_interleaved_abc.py`](https://github.com/sanderwood/clamp3/blob/main/preprocessing/abc/batch_interleaved_abc.py):  
   ```bash
   python batch_interleaved_abc.py <input_dir> <output_dir>
   ```
   - **Input:** Directory containing `.abc` (Standard ABC) files  
   - **Output:** Directory where Interleaved ABC files will be saved *(for CLaMP 3 use)*  

##### **1.2 Convert MIDI to MTF Format**  

CLaMP 3 processes performance signals in **MIDI Text Format (MTF)**. Convert **MIDI files** (`.mid`, `.midi`) into **MTF format** using [`batch_midi2mtf.py`](https://github.com/sanderwood/clamp3/blob/main/preprocessing/midi/batch_midi2mtf.py):  
```bash
python batch_midi2mtf.py <input_dir> <output_dir> --m3_compatible
```
- **Input:** Directory containing `.mid`, `.midi` files  
- **Output:** Directory where `.mtf` files will be saved *(MTF format for CLaMP 3)*  
- **Important:** The `--m3_compatible` flag **must be included** to ensure the output format is compatible with CLaMP 3. Without this flag, the extracted MTF files **will not work** correctly in the pipeline.

##### **1.3 Extract Audio Features using MERT**
For audio processing, CLaMP 3 uses **MERT-extracted features** instead of raw waveforms. Extract MERT-based features from raw audio (`.mp3`, `.wav`) using [`extract_mert.py`](https://github.com/sanderwood/clamp3/blob/main/preprocessing/audio/extract_mert.py):

```bash
python extract_mert.py --input_path <input_path> --output_path <output_path> --model_path m-a-p/MERT-v1-95M --mean_features
```
- **Input:** `.mp3`, `.wav`  
- **Output:** `.npy` *(Processed audio features for CLaMP 3)*  

### **Training and Feature Extraction**  

#### **1. Training Models**  
CLaMP 3 is the most powerful music retrieval model, and in most cases, retraining is not needed. However, if necessary, follow these steps.  

1. Modify **[config.py](https://github.com/sanderwood/clamp3/blob/main/code/config.py)** to adjust **hyperparameters** and **data paths**.  

2. Train on your own data.

To train CLaMP 3 on **symbolic music** (e.g., sheet music, MIDI), run:  
```bash
python -m torch.distributed.launch --nproc_per_node=<GPUs> --use_env train_clamp3_symbolic.py
```
For **audio data**, use:  
```bash
python -m torch.distributed.launch --nproc_per_node=<GPUs> --use_env train_clamp3_audio.py
```

##### **Using Pre-Trained Models (Recommended)**  
For most use cases, it's best to use pre-trained weights instead of training from scratch.  

| Version | Best for | Download Link |
|---------|---------|--------------|
| **CLaMP 3 SAAS** | **Audio-based retrieval (Recommended)** | [Download SAAS](https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth) |
| **CLaMP 3 C2** | **Symbolic music retrieval (Sheet music, MIDI)** | [Download C2](https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_c2_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth) |

##### **How to Switch Between Versions?**  
By default, CLaMP 3 is configured for the **SAAS version** (optimized for audio).  
- If working with **symbolic music (MIDI, sheet music)**, use the **C2 version**:  
  **Modify line 66 in `config.py`** from `"saas"` to `"c2"`.
  
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

CLaMP 3 enables **semantic search** by computing feature similarities between a query and a reference dataset. This search can be performed across different modalities, including **text, music, and images**.  

To run a semantic search, use the [`clamp3_search.py`](https://github.com/sanderwood/clamp3/blob/main/inference/clamp3_search.py) script:

```bash
python clamp3_search.py <query_file> <reference_folder> [--top_k TOP_K]
```
- **`<query_file>`**: Path to the query file (`.txt`, `.png`, `.jpg`, `.mid`, `.mxl`, `.wav`, etc.).  
- **`<reference_folder>`**: Folder containing reference files (music, images, or text).  
- **`--top_k`**: *(Optional)* Number of top similar items to display (default is `10`).  

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
If you find CLaMP 3 useful in your work, please consider citing our paper:

```bibtex
@misc{wu2025clamp3universalmusic,
  title={CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages}, 
  author={Shangda Wu and Zhancheng Guo and Ruibin Yuan and Junyan Jiang and Seungheon Doh and Gus Xia and Juhan Nam and Xiaobing Li and Feng Yu and Maosong Sun},
  year={2025},
  eprint={2502.10362},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2502.10362}
}
```