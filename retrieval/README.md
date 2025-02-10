# Retrieval Codebase

## Overview  
This codebase provides scripts for evaluating model retrieval performance, performing semantic searches, and calculating similarity metrics using CLaMP3-extracted **global** feature vectors from music or text data.

## Repository Structure  
The [retrieval/](https://github.com/sanderwood/clamp3/tree/main/retrieval) folder contains the following scripts:

### 1. [clamp3_score.py](https://github.com/sanderwood/clamp3/blob/main/retrieval/clamp3_score.py)  
This script calculates cosine similarity between average feature vectors from two sets of `.npy` files, serving as a measure of similarity between reference and test datasets.

**Usage:**
```bash
python clamp3_score.py <reference_folder> <test_folder>
```
- `reference_folder`: Folder containing reference `.npy` files.
- `test_folder`: Folder containing test `.npy` files.

**Functionality:**
- Computes the average feature vector for each folder.
- Calculates the cosine similarity between the averaged vectors.
- Outputs the similarity score (rounded to four decimal places).

### 2. [semantic_search.py](https://github.com/sanderwood/clamp3/blob/main/retrieval/semantic_search.py) 
Performs semantic search by calculating the cosine similarity between a query feature and a set of stored features.

**Usage:**
```bash
python semantic_search.py <query_file> <features_folder> [--top_k TOP_K]
```
- `query_file`: Path to the query feature file (e.g., `ballad.npy`).
- `features_folder`: Folder containing feature files for comparison.
- `--top_k`: (Optional) Number of top similar items to display (default is 10).

**Functionality:**
- Loads the query feature and feature vectors from the given folder.
- Computes cosine similarity between the query and each feature.
- Displays the top K most similar features and their similarity scores.

### 3. [eval_mrr.py](https://github.com/sanderwood/clamp3/blob/main/retrieval/eval_mrr.py) 
Calculates evaluation metrics for semantic search by comparing query features to reference features.

**Usage:**
```bash
python eval_mrr.py <query_folder> <reference_folder>
```
- `query_folder`: Folder containing query features (`.npy` files).
- `reference_folder`: Folder containing reference features (`.npy` files).

**Functionality:**
- Loads query and reference features from the specified folders.
- Computes the following metrics based on cosine similarity:
  - **Mean Reciprocal Rank (MRR)**
  - **Hit@1**
  - **Hit@10**
  - **Hit@100**
- Outputs the metrics to the console.