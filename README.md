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

---

## **Overview**
CLaMP 3 is a **state-of-the-art** framework for **music information retrieval (MIR)** across multiple **modalities** (Text, 🎵 Audio, 🎼 Sheet Music, 🎹 Performance MIDIs) and **languages** (🌐 27 trained, 100 supported). It leverages **contrastive learning** to align diverse music formats into a **shared representation space**, enabling seamless cross-modal retrieval. You can think of it as an advanced **CLAP/MuLan** for music.

🚀 **Why CLaMP 3?**  
✅ **Multimodal**: Works with 🎼 **sheet music**, 🎵 **audio**, 🎹 **MIDI performance**  
✅ **Multilingual**: Supports **27 trained** & generalizes to **100 languages**  
✅ **SOTA Performance**: Outperforms previous baselines in **MIR**  

---

## ✨ **Key Features**  

### **Multimodal Support**  
- **Sheet Music**: Interleaved ABC notation (**512 bars**)  
- **Performance Signals**: MIDI Text Format (**512 MIDI messages**)  
- **Audio Recordings**: [MERT](https://arxiv.org/abs/2306.00107) features (**640 sec of audio**)  

### **Multilingual Capabilities**  
- Trained on **27 languages**, generalizes to **100+** using [XLM-R](https://arxiv.org/abs/1911.02116)  

### **Datasets & Benchmarks**  
- **[M4-RAG](https://huggingface.co/datasets/sander-wood/m4-rag)**: **2.31M music-text pairs** 🌎  
- **[WikiMT-X](https://huggingface.co/datasets/sander-wood/wikimt-x)**: **1,000 music triplets**  

---

## 🔥 **What Can CLaMP 3 Do?**  

💡 **Text-to-Music Retrieval**: Search music with text (100 languages!)  
📸 **Image-to-Music Retrieval**: Match music to images 🎨  
🔄 **Cross-Modal Retrieval**: Find related music across formats  
🛠️ **Zero-Shot Classification**: Identify genre, mood, & style 🏷️  
🎼 **Semantic Similarity**: Measure similarity between generated & reference music  

👉 **Try it out**: [CLaMP 3 Homepage](https://sanderwood.github.io/clamp3/)  

---

## **Quick Start Guide**  
For users who want to get started quickly without delving into the details, follow these steps:

### **Install Environment**  
```bash
conda create -n clamp3 python=3.10.16 -y
conda activate clamp3
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

### **Overview of `clamp3_*.py` Scripts**  
CLaMP 3 provides the `clamp3_*.py` script series for **streamlined data preprocessing, feature extraction, retrieval, similarity computation, and evaluation**. These scripts offer an easy-to-use solution for processing different modalities with minimal configuration.

**Common Features of `clamp3_*.py` Scripts:**  
- **End-to-End Processing**: Each script handles the entire pipeline in a single command.  
- **Automatic Modality Detection**:  
  Simply specify the file path, and the script will automatically detect the modality (e.g., **audio**, **performance signals**, **sheet music**, **images**, or **text**) and extract the relevant features. Supported formats include:
  - **Audio**: `.mp3`, `.wav`
  - **Performance Signals**: `.mid`, `.midi`
  - **Sheet Music**: `.mxl`, `.musicxml`, `.xml`
  - **Images**: `.png`, `.jpg`
  - **Text**: `.txt`
- **First-Time Model Download**:  
  - The necessary model weights for **[CLaMP 3 (SAAS)](https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth)**, **[MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M)**, and any other required models will be automatically downloaded if needed.
  - Once downloaded, models are cached and will not be re-downloaded in future runs.  

- **Feature Management**:  
  - Extracted features are saved in `inference/` and **won't be overwritten** to avoid redundant computations.  
  - **To run retrieval on a new dataset**, manually delete the corresponding folder inside `inference/` (e.g., `inference/audio_features/`). Otherwise, previously extracted features will be reused.  
  - Temporary files are stored in `temp/` and **are cleaned up after each run**.  

> **Note**: All files within a folder must belong to the same modality; the script will process them based on the first detected format.

#### **[`clamp3_search.py`](https://github.com/sanderwood/clamp3/blob/main/clamp3_search.py) - Running Retrieval Tasks**

This script performs semantic retrieval tasks, comparing a query file to reference files in `ref_dir`. Typically, the larger and more diverse the files in `ref_dir`, the better the chances of finding a semantically matching result.

```bash
python clamp3_search.py <query_file> <ref_dir> [--top_k TOP_K]
```

- **Text-to-Music Retrieval**:  
  Query is a `.txt` file, and `ref_dir` contains music files. Retrieves the music most semantically similar to the query text.
  
- **Image-to-Music Retrieval**:  
  Query is an image (`.png`, `.jpg`), and `ref_dir` contains music files. **BLIP** generates a caption for the image to find the most semantically matching music.

- **Music-to-Music Retrieval**:  
  Query is a music file, and `ref_dir` contains music files (same or different modality). Supports **cross-modal retrieval** (e.g., retrieving audio using sheet music).

- **Zero-Shot Classification**:  
  Query is a music file, and `ref_dir` contains **text-based class prototypes** (e.g., `"It is classical"`, `"It is jazz"`). The highest similarity match is the classification result.

- **Optional `--top_k` Parameter**:  
  You can specify the number of top results to retrieve using the `--top_k` argument. If not provided, the default value is 10.  
  Example:
  ```bash
  python clamp3_search.py <query_file> <ref_dir> --top_k 3
  ```

  **Example Output**:
  ```
  Top 3 results among 1000 candidates:
  4tDYMayp6Dk 0.7468
  vGJTaP6anOU 0.7333
  JkK8g6FMEXE 0.7054
  ```

#### **[`clamp3_score.py`](https://github.com/sanderwood/clamp3/blob/main/clamp3_score.py) - Semantic Similarity Calculation**

This script compares files in a query directory to a reference directory. By default, it uses **group mode**, but you can switch to **pairwise mode** for paired data.

```bash
python clamp3_score.py <query_dir> <ref_dir> [--pairwise]
```

- **Group Mode (default)**:  
  Compares all query files to all reference files and calculates the average similarity. **Use when you don't have paired data** or when dealing with large datasets.  

  **Example**:  
  To compare generated music to ground truth music files (no pairs available), use **group mode**.

  ```bash
  python clamp3_score.py query_dir ref_dir
  ```

  **Example Output (Group Mode)**:
  ```
  Total query features: 1000
  Total reference features: 1000
  Group similarity: 0.6711
  ```

- **Pairwise Mode**:  
  Compares query files with their corresponding reference files based on **same prefix** (before the dot) and **identical folder structure**. **Use when you have paired data** and the dataset is of manageable size (e.g., thousands of pairs).  

  **Example**:  
  To evaluate a **text-to-music generation model**, where each prompt (e.g., `sample1.txt`) corresponds to one or more generated music files (e.g., `sample1.1.wav`, `sample1.2.wav`), use **pairwise mode**.

  ```bash
  python clamp3_score.py query_dir ref_dir --pairwise
  ```

  **Folder structure**:
  ```
  query_dir/
  ├── en/
  │   ├── sample1.wav  
  ├── zh/
  │   ├── sample1.1.wav   
  │   ├── sample1.2.wav    
  │   ├── sample2.wav  

  ref_dir/
  ├── en/
  │   ├── sample1.txt
  ├── zh/
  │   ├── sample1.txt
  │   ├── sample2.txt
  ```

  - Files with the **same prefix** (e.g., `query_dir/en/sample1.wav` and `ref_dir/en/sample1.txt`) are treated as pairs.
  - Multiple query files (e.g., `query_dir/zh/sample1.1.wav`, `query_dir/zh/sample1.2.wav`) can correspond to one reference file (e.g., `query_dir/zh/sample1.txt`).

  **Example Output (Pairwise Mode)**:
  ```
  Total query features: 1000
  Total reference features: 1000
  Avg. pairwise similarity: 0.1639
  ```

  In **pairwise mode**, the script will additionally output a JSON Lines file (`inference/pairwise_similarities.jsonl`) with the similarity scores for each query-reference pair.  
  For example:
  ```json
  {"query": "txt_features/UzUybLGvBxE.npy", "reference": "mid_features/UzUybLGvBxE.npy", "similarity": 0.2289600819349289}
  ```

  > **Note**: The file paths in the output will retain the folder structure and file names, but the top-level folder names and file extensions will be replaced.

#### **[`clamp3_eval.py`](https://github.com/sanderwood/clamp3/blob/main/clamp3_eval.py) - Evaluating Retrieval Performance**

This script evaluates **CLaMP3's retrieval performance on a paired dataset**, measuring how accurately the system ranks the correct reference files for each query using metrics like **MRR** and **Hit@K**.

```bash
python clamp3_eval.py <query_dir> <ref_dir>
```

- **Matching Folder Structure & Filenames**:  
  Requires paired query and reference files, with identical folder structure and filenames between `query_dir` and `ref_dir`. This matches the requirements of **pairwise mode** in `clamp3_score.py`.

- **Evaluation Metrics**:  
  The script calculates the following retrieval metrics:  
  - **MRR (Mean Reciprocal Rank)**  
  - **Hit@1**, **Hit@10**, and **Hit@100**  

**Example Output**:
```
Total query features: 1000
Total reference features: 1000
MRR: 0.3301
Hit@1: 0.251
Hit@10: 0.482
Hit@100: 0.796
```

- **Additional Output**:  
  A JSON Lines file (`inference/retrieval_ranks.jsonl`) with query-reference ranks:
  ```json
  {"query": "txt_features/HQ9FaXu55l0.npy", "reference": "xml_features/HQ9FaXu55l0.npy", "rank": 6}
  ```

## **Repository Structure**
- **[code/](https://github.com/sanderwood/clamp3/tree/main/code)** → Training & feature extraction scripts.
- **[classification/](https://github.com/sanderwood/clamp3/tree/main/classification)** → Linear classification training and prediction.  
- **[inference/](https://github.com/sanderwood/clamp3/tree/main/inference)** → Semantic search, similarity calculations, and retrieval evaluation.  
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
accelerate launch extract_clamp3.py --epoch <epoch> <input_dir> <output_dir> --get_global
```
- **`--epoch <epoch>`:** (Optional) Specify the checkpoint epoch.  
- **`<input_dir>`:** Directory containing the input files.  
- **`<output_dir>`:** Destination folder for the output `.npy` features.  
- **`--get_global`**: **(Required for retrieval!)** Extracts a **global semantic vector** for each input.  

All extracted features are stored as `.npy` files.

> **Note**: For retrieval, `--get_global` must be used. Without it, CLaMP 3 will not work correctly for retrieval tasks. You only omit `--get_global` if you are performing downstream fine-tuning or need raw feature extraction for custom tasks.

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
