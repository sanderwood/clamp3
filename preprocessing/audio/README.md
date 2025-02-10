# Audio Feature Extraction

This code is based on the [MARBLE-Benchmark](https://github.com/a43992899/MARBLE-Benchmark/tree/ffcf8fdd1d465e20a61a31ddf649f4261873d32d/benchmark/models/musichubert_hf) source code.

## Setup Instructions

1. **Create and activate a Conda environment**:
   ```bash
   conda create -n marble python=3.8
   conda activate marble
   pip install -r requirements.txt
   ```

2. **Download the HuBERT model** from Hugging Face:
   - [Download Model](https://huggingface.co/m-a-p/MERT-v1-95M)

3. **Extract audio features**:
   ```bash
   python extract_mert.py --input_path <input_path> --output_path <output_path> --model_path musichubert_hf/MERT-v1-95M --mean_features
   ```

