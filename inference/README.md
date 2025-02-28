# **Inference Codebase**  
This codebase provides scripts for **retrieval, similarity computation, and evaluation** using **CLaMP3-extracted feature vectors** from music, text, or images.

## **Repository Structure**  
The [inference/](https://github.com/sanderwood/clamp3/tree/main/inference) folder contains the following scripts:

### **1. [`clamp3_score.py`](https://github.com/sanderwood/clamp3/blob/main/inference/clamp3_score.py) - Computing Semantic Similarity**  
This script calculates the **semantic similarity** between two datasets by computing cosine similarity between their **feature vectors**.

```bash
python clamp3_score.py <query_folder> <reference_folder> [--group]
```
- **`<query_folder>`**: Folder containing query `.npy` feature files.  
- **`<reference_folder>`**: Folder containing reference `.npy` feature files.  
- **`--group`**: *(Optional)* Computes similarity between all query and reference features and returns an average similarity score.  

#### **Modes of Operation**  

- **Pairwise Mode (default)**:  
  Matches **query and reference files based on identical folder structure and filename prefix** (before the dot) and computes their similarity.  
  **Use for paired datasets.**  
  ```bash
  python clamp3_score.py query_folder reference_folder
  ```
  **Example Output (Pairwise Mode)**:
  ```bash
  Total query features: 1000
  Total reference features: 1000
  Avg. pairwise similarity: 0.1639
  ```

  In **pairwise mode**, results are also saved to a JSON Lines file (`inference/pairwise_similarities.jsonl`) for each pair:  
  ```json
  {"query": "txt_features/UzUybLGvBxE.npy", "reference": "mid_features/UzUybLGvBxE.npy", "similarity": 0.2289600819349289}
  ```

- **Group Mode**:  
  Computes similarity between **all query and reference features** and returns an average similarity score.  
  **Use for large-scale comparisons (fast).**  
  ```bash
  python clamp3_score.py query_folder reference_folder --group
  ```
  **Example Output (Group Mode)**:
  ```bash
  Total query features: 1000
  Total reference features: 1000
  Group similarity: 0.6711
  ```

### **2. [`clamp3_search.py`](https://github.com/sanderwood/clamp3/blob/main/inference/clamp3_search.py) - Running Retrieval Tasks**  
This script performs **semantic retrieval**, ranking reference files by cosine similarity to a query feature.

```bash
python clamp3_search.py <query_file> <reference_folder> [--top_k TOP_K]
```
- **`<query_file>`**: Path to the query feature (`.npy`).  
- **`<reference_folder>`**: Folder containing reference features (`.npy`).  
- **`--top_k`**: *(Optional)* Number of top similar items to display (default is `10`).  

#### **Example Output**:
```
Top 3 results among 1000 candidates:
4tDYMayp6Dk 0.7468
vGJTaP6anOU 0.7333
JkK8g6FMEXE 0.7054
```

### **3. [`clamp3_eval.py`](https://github.com/sanderwood/clamp3/blob/main/inference/clamp3_eval.py) - Evaluating Retrieval Performance**  
This script evaluates **CLaMP3's retrieval performance on a paired dataset**, measuring how accurately the system ranks the correct reference files for each query.

```bash
python clamp3_eval.py <query_folder> <reference_folder>
```
- **`<query_folder>`**: Folder containing query feature files (`.npy`).  
- **`<reference_folder>`**: Folder containing reference feature files (`.npy`).  

#### **Evaluation Metrics**  
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
  The script also generates a JSON Lines file (`inference/retrieval_ranks.jsonl`) with query-reference ranks:
  ```json
  {"query": "txt_features/HQ9FaXu55l0.npy", "reference": "xml_features/HQ9FaXu55l0.npy", "rank": 6}
  ```

### **4. [`image_captioning.py`](https://github.com/sanderwood/clamp3/blob/main/inference/image_captioning.py) - Generating Image Captions using BLIP**  
This script generates descriptive captions for images using the **BLIP** model, which can then be used for tasks like **image-to-music retrieval**.

```bash
python image_captioning.py <input_dir> <output_dir>
```
- **`<input_dir>`**: Path to the input directory containing image files (e.g., `.jpg`, `.png`).  
- **`<output_dir>`**: Path to the output directory where captions will be saved as text files.