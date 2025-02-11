# **Audio Feature Extraction**
This folder provides scripts for extracting **MERT-based audio features**—the representation used by **CLaMP 3’s audio encoder**. These features are generated using the **MERT-v1-95M model**, which processes audio into **5-second non-overlapping segments** and averages across all layers and time steps to produce a **single feature per segment**.

## **1. Create and Activate a Conda Environment**
Set up your environment by running:
```bash
conda create -n mert-extraction python=3.8
conda activate mert-extraction
pip install -r requirements.txt
```

## **2. Download the MERT Model**
Download [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) model from Hugging Face.

## **3. [extract_mert.py](https://github.com/sanderwood/clamp3/blob/main/preprocessing/audio/extract_mert.py)**
**Step 1:** Extracts **MERT features** from audio files.

- **Execution:**  
  Run the script using the following command:
  ```bash
  python extract_mert.py --input_path <input_path> --output_path <output_path> --model_path m-a-p/MERT-v1-95M --mean_features
  ```
- **Input:** Audio files (`.mp3`, `.wav`).
- **Output:** MERT-extracted features (`.npy`).
