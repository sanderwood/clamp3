# CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages

<p align="center">
  <img src="overview.png" alt="CLaMP 3 Overview" width="50%">
</p>


## Overview
CLaMP 3 is a unified framework for cross-modal and cross-lingual music information retrieval (MIR). By using contrastive learning, it aligns sheet music, audio, performance signals, and multilingual text into a shared representation space, enabling retrieval across unaligned musical modalities. Key features include:

- **Multimodal Support:**
   1. **Sheet Music:** Uses Interleaved ABC notation.
   2. **Performance Signals:** Processes MIDI Text Format (MTF) data.
   3. **Audio Recordings:** Works with audio features extracted by MERT.

- **Multilingual Capabilities:** Supports 100 languages ([XLM-R](https://arxiv.org/abs/1911.02116)) and generalizes effectively beyond its 27-language training data.

- **Dataset and Benchmark:**
  - Trained on **M4-RAG**, a large-scale dataset of 2.31M high-quality music-text pairs across 27 languages and 194 countries.
  - Introduces **WikiMT-X**, a benchmark containing 1,000 triplets of sheet music, audio, and text.

CLaMP 3 achieves state-of-the-art performance across multiple MIR tasks, advancing research in multimodal and multilingual music systems.

### Links
- CLaMP 3 Demo Page (Coming Soon...)
- CLaMP 3 Paper (Coming Soon...)
- [CLaMP 3 Code](https://github.com/sanderwood/clamp3)
- [CLaMP 3 Model Weights](https://huggingface.co/sander-wood/clamp3/tree/main)
- [M4-RAG Pre-training Dataset](https://huggingface.co/datasets/sander-wood/m4-rag)
- [WikiMT-X Evaluation Benchmark](https://huggingface.co/datasets/sander-wood/wikimt-x)

> **Note** Ensure the model weights for CLaMP 3 are placed under the `code/` folder for proper loading. Also, verify that the configuration hyperparameters are correctly set.

## Repository Structure
- **`code/`**: Contains scripts for training CLaMP 3 and extracting features from music and text data. You can modify hyperparameters and file paths in the configuration files for custom training.
  
- **`classification/`**: Includes scripts for classification tasks using extracted features, such as training linear classification models and making predictions.

- **`processing/`**: Scripts for converting your data into compatible formats (interleaved ABC notation, MTF, or MERT-extracted audio features). These are required for CLaMP 3 to work with the data.

- **`retrieval/`**: Provides scripts for evaluating model performance, conducting semantic searches, and calculating similarity metrics based on extracted feature vectors.

## Getting Started
### Environment Setup
To set up the environment for CLaMP 3, run the following commands:

```bash
conda env create -f environment.yml
conda activate clamp3
```

### Data Preparation
1. **Convert Files**: Navigate to the `preprocessing/` folder and convert your music files into a compatible format (interleaved ABC notation, MTF, or MERT-extracted audio features) suitable for use with CLaMP 3. Whether you are training or performing inference, **you must use these preprocessing scripts to ensure the data is in the correct format**.
   1. **Interleaved ABC Notation**:
      - Convert MusicXML files to ABC using `/preprocessing/abc/batch_xml2abc.py`.
      - Process the ABC files into interleaved notation using `/preprocessing/abc/batch_interleaved_abc.py`.
   2. **MTF**:
      - Convert MIDI files to MTF format using `/preprocessing/midi/batch_midi2mtf.py`.
   3. **MERT-extracted Audio Features**:
      - Extract audio features using MERT by running the scripts in the `/preprocessing/audio/` folder. These features will be saved as `.npy` files and are ready for use in CLaMP 3.

2. **Prepare Text Metadata (Optional)**: If you plan to train the model, you will need to prepare corresponding metadata for each music file. The metadata should be in JSON format, containing details like title, artist, region, language, and description.

   Example:
   ```json
   {
       "filepaths": ["audio/--/---aL9TdeI4.npy"],
       "id": "---aL9TdeI4",
       "title": "Mairi's Wedding",
       "artists": ["Noel McLoughlin"],
       "region": "United Kingdom of Great Britain and Northern Ireland",
       "language": "English",
       "genres": ["Folk", "Traditional"],
       "tags": ["Scottish", "Wedding", "Traditional", "Folk", "Celtic"],
       "background": "Mairi's Wedding is a Scottish folk song...",
       "analysis": "The song has a lively and upbeat Scottish folk rhythm...",
       "description": "A traditional folk song with a joyful celebration...",
       "scene": "The setting is a picturesque Scottish village on a sunny morning...",
       "translations": { "language": "Vietnamese", "background": "Bài hát \"Đám Cưới Mairi\"..." }
   }
   ```

   Once your JSON files are ready, merge them into a single `.jsonl` file and structure the directories as shown:

   ```
   /your-data-folder/
   ├── abc/
   ├── audio/
   ├── mtf/
   ├── merged_output.jsonl
   ```

### Training and Feature Extraction
2. **Training Models**: If you want to train CLaMP 3, check the training scripts in the `code/` folder and modify the `config.py` file to set your hyperparameters and data paths.

3. **Extracting Features**: After training or with pre-trained weights, extract features using `code/extract_clamp3.py`. The script automatically detects the modality based on the file extension (e.g., `.txt`, `.abc`, `.mtf`, `.npy`).

### Classification and Retrieval
4. **Classification**: To perform classification on the extracted features, navigate to the `classification/` directory. You’ll find scripts for training and making predictions using linear classification models.

5. **Semantic Search**: To conduct semantic searches using the extracted features, refer to the scripts in the `retrieval/` folder.

## Citation
Coming Soon...
<!-- If you use CLaMP 3, M4-RAG, or WikiMT-X in your research, please cite the following paper:

bibtex
@misc{wu2024clamp2multimodalmusic,
      title={CLaMP 2: Multimodal Music Information Retrieval Across 101 Languages Using Large Language Models}, 
      author={Shangda Wu and Yashan Wang and Ruibin Yuan and Zhancheng Guo and Xu Tan and Ge Zhang and Monan Zhou and Jing Chen and Xuefeng Mu and Yuejie Gao and Yuanliang Dong and Jiafeng Liu and Xiaobing Li and Feng Yu and Maosong Sun},
      year={2024},
      eprint={2410.13267},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2410.13267}, 
} -->
