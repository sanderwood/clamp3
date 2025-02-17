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

> **Note:** When evaluating generated music, ensure that the generated and reference music share similar distributions (e.g., both are jazz pieces), to minimize confounding factors and get a more accurate evaluation.

### **2. [semantic_search.py](https://github.com/sanderwood/clamp3/blob/main/retrieval/semantic_search.py)**
This script performs semantic search by calculating the cosine similarity between a query feature and a set of reference features.

```bash
python semantic_search.py <query_file> <reference_folder> [--top_k TOP_K]
```
- **`<query_file>`**: Path to the query feature (e.g., `ballad.npy`).
- **`<reference_folder>`**: Folder containing reference features for comparison.
- **`--top_k`**: *(Optional)* Number of top similar items to display (default is 10).

CLaMP 3's semantic search enables various retrieval and evaluation tasks by comparing features extracted from queries and reference data. Generally, the larger and more diverse the reference music dataset, the higher the likelihood of retrieving relevant and accurately matched music.

##### **1. Text-to-Music Retrieval**  
- **Query:** Text description of the desired music.  
- **Reference:** Music data (e.g., audio files).  
- **Output:** Retrieves music that best matches the semantic meaning of the text description.

##### **2. Image-to-Music Retrieval**  
- **Query:** Generate an image caption using models like [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base).  
- **Reference:** Music data (e.g., audio files). 
- **Output:** Finds music that semantically aligns with the image.

##### **3. Cross-Modal and Same-Modal Music Retrieval**  
- **Cross-Modal Retrieval:**  
  - **Query:** Music data from one modality (e.g., audio).  
  - **Reference:** Music data from another modality (e.g., MIDI, ABC notation).  
  - **Output:** Finds semantically similar music across different representations.

- **Same-Modal Retrieval (Semantic-Based Music Recommendation):**  
  - **Query & Reference:** Both are from the same modality (e.g., audio-to-audio).  
  - **Output:** Recommends similar music based on semantic meaning.

##### **4. Zero-Shot Music Classification**  
- **Query:** Music data.  
- **Reference:** Class descriptions (e.g., "It is classical," "It is folk").  
- **Output:** Assigns the most relevant class based on feature similarity.

##### **5. Music Semantic Similarity Evaluation**  
- **Query:** High-quality music or music generation prompt.  
- **Reference:** Generated music.  
- **Output:** Ranks generated music based on semantic similarity to the query. For large-scale evaluation between generated music and reference music, it is recommended to use [`clamp3_score.py`](https://github.com/sanderwood/clamp3/blob/main/retrieval/clamp3_score.py).

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