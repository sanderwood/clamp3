# **Retrieval Codebase**
This codebase provides scripts for evaluating model retrieval performance, performing semantic searches, and calculating similarity metrics using CLaMP 3-extracted **global** feature vectors from music or text data.

## **Repository Structure**  
The [retrieval/](https://github.com/sanderwood/clamp3/tree/main/retrieval) folder contains the following scripts:

### **1. [clamp3_score.py](https://github.com/sanderwood/clamp3/blob/main/retrieval/clamp3_score.py)**
This script calculates cosine similarity between average feature vectors from two sets of `.npy` files, serving as a measure of similarity between test and reference datasets.

```bash
python clamp3_score.py <test_folder> <reference_folder>
```
- **`test_folder`**: Folder containing test `.npy` files.
- **`reference_folder`**: Folder containing reference `.npy` files.

### **2. [semantic_search.py](https://github.com/sanderwood/clamp3/blob/main/retrieval/semantic_search.py)**
This script performs semantic search by calculating the cosine similarity between a query feature and a set of reference features.

```bash
python semantic_search.py <query_file> <reference_folder> [--top_k TOP_K]
```
- **`<query_file>`**: Path to the query feature (e.g., `ballad.npy`).
- **`<reference_folder>`**: Folder containing reference features for comparison.
- **`--top_k`**: *(Optional)* Number of top similar items to display (default is 10).

### **3. [eval_mrr.py](https://github.com/sanderwood/clamp3/blob/main/retrieval/eval_mrr.py)**
This script calculates evaluation metrics for semantic search by comparing query features to reference features.

```bash
python eval_mrr.py <query_folder> <reference_folder>
```
- **`<query_folder>`**: Folder containing query features (`.npy` files).
- **`<reference_folder>`**: Folder containing reference features (`.npy` files).
- **Metric Computation**: Computes evaluation metrics based on cosine similarity:
  - **Mean Reciprocal Rank (MRR)**
  - **Hit@1**
  - **Hit@10**
  - **Hit@100**