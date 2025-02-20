# **Inference Codebase**  
This codebase provides scripts for running retrieval tasks, evaluating model performance, and computing similarity metrics using **CLaMP 3-extracted feature vectors** from music, text, or images.

## **Repository Structure**  
The [inference/](https://github.com/sanderwood/clamp3/tree/main/inference) folder contains the following scripts:

### **1. [clamp3_score.py](https://github.com/sanderwood/clamp3/blob/main/inference/clamp3_score.py)**  
This script computes the **semantic similarity** between two datasets using cosine similarity between their **average feature vectors**.

```bash
python clamp3_score.py <query_folder> <reference_folder>
```
- **`<query_folder>`**: Folder containing query `.npy` feature files.  
- **`<reference_folder>`**: Folder containing reference `.npy` feature files.  

> **Use Case:** This is typically used for evaluating **the overall similarity between a generated music dataset and a ground truth dataset**.

### **2. [clamp3_search.py](https://github.com/sanderwood/clamp3/blob/main/inference/clamp3_search.py)**  
This script **performs retrieval tasks** by calculating cosine similarity between query and reference feature vectors.

```bash
python clamp3_search.py <query_file> <reference_folder> [--top_k TOP_K]
```
- **`<query_file>`**: Path to the query feature (`.npy`).  
- **`<reference_folder>`**: Folder containing reference features (`.npy`).  
- **`--top_k`**: *(Optional)* Number of top similar items to display (default is `10`).  

CLaMP 3 enables multiple retrieval tasks based on **semantic similarity** across modalities:

##### **1. Text-to-Music Retrieval**  
- **Query:** A `.txt` file containing a text description.  
- **Reference:** Music data (e.g., audio, MIDI).  
- **Output:** Retrieves music that best matches the semantic meaning of the text.  

##### **2. Image-to-Music Retrieval**  
- **Query:** An image (`.png`, `.jpg`).  
- **Reference:** Music data.  
- **Output:** Finds music that semantically aligns with the **BLIP-generated caption** of the image.  

##### **3. Cross-Modal and Same-Modal Music Retrieval**  
- **Cross-Modal Retrieval:**  
  - **Query:** A music file in one modality (e.g., MIDI).  
  - **Reference:** Music files in another modality (e.g., sheet music, audio).  
  - **Output:** Finds semantically similar music across different representations.  

- **Same-Modal Retrieval (Semantic-Based Music Recommendation):**  
  - **Query & Reference:** Both are from the same modality (e.g., audio-to-audio).  
  - **Output:** Recommends similar music based on feature similarity.  

##### **4. Zero-Shot Music Classification**  
- **Query:** A music file.  
- **Reference:** Text-based class descriptions (e.g., `"It is classical"`, `"It is jazz"`).  
- **Output:** Assigns the most relevant class based on feature similarity.  

##### **5. Music Semantic Similarity Evaluation**  
- **Query:** A text/music prompt.  
- **Reference:** Generated music.  
- **Output:** Ranks generated music based on semantic similarity to the query.  
- For large-scale evaluation, use [`clamp3_score.py`](https://github.com/sanderwood/clamp3/blob/main/inference/clamp3_score.py).  

### **3. [clamp3_eval.py](https://github.com/sanderwood/clamp3/blob/main/inference/clamp3_eval.py)**  
This script computes **retrieval evaluation metrics** by comparing query features to reference features.

```bash
python clamp3_eval.py <query_folder> <reference_folder>
```
- **`<query_folder>`**: Folder containing query feature files (`.npy`).  
- **`<reference_folder>`**: Folder containing reference feature files (`.npy`).  
- **Metrics Computed:**  
  - **Mean Reciprocal Rank (MRR)**  
  - **Hit@1, Hit@10, Hit@100**  
  - **Average Pair Similarity**  

> **Example Folder Structure:**  
  ```
  query_folder/  
  ├── en/  
  │   ├── sample1.txt  
  │   ├── sample2.txt  
  ├── zh/  
  │   ├── sample1.txt  
  ```
  ```
  reference_folder/  
  ├── en/  
  │   ├── sample1.wav  
  │   ├── sample2.wav  
  ├── zh/  
  │   ├── sample1.wav  
  ```
  - Here, `query_folder/en/sample1.txt` is matched with `reference_folder/en/sample1.wav`.  
  - Ensures a **one-to-one correspondence** for proper evaluation.  

> **Note:** Unlike `clamp3_search.py`, which is used for **retrieval tasks**, `clamp3_eval.py` is used for **evaluating retrieval quality** and computing ranking-based performance metrics.