# **Music Classification Codebase**
This codebase provides scripts for training and using a **linear classification model** on music-related feature representations. It supports both training a classifier from extracted features and performing inference on new data.

## **Repository Structure**  
The [classification/](https://github.com/sanderwood/clamp3/tree/main/classification) folder contains the core scripts:

### **1. [train_cls.py](https://github.com/sanderwood/clamp3/blob/main/classification/train_cls.py)**
This script trains a linear classification model on the extracted feature representations.

#### **Arguments**
- **`--train_folder`**: Path to the training folder.
- **`--eval_folder`**: Path to the evaluation folder. If not specified, 20% of the training data will be randomly set aside for evaluation.
- **`--eval_split`**: Fraction of training data used for evaluation (default: `0.2`).
- **`--wandb_key`**: Weights & Biases API key for logging.
- **`--num_epochs`**: Maximum number of training epochs (default: `1000`).
- **`--learning_rate`**: Learning rate for optimization (default: `1e-5`).
- **`--balanced_training`**: Flag to balance labels in the training data.
- **`--wandb_log`**: Flag to log training metrics to Weights & Biases.

#### **Usage**  
Run the training script using:
```bash
python train_cls.py --train_folder <path> --eval_folder <path> [--eval_split <float>] [--num_epochs <int>] [--learning_rate <float>] [--balanced_training] [--wandb_log]
```
**Example:**
```bash
python train_cls.py --train_folder data/train
```

### **2. [inference_cls.py](https://github.com/sanderwood/clamp3/blob/main/classification/inference_cls.py)**
This script performs classification on extracted feature vectors using a trained linear probe model.

#### **JSON Output Format**  
The classification results are saved in a JSON file with the following structure:
```json
{
    "path/to/feature1.npy": "class_A",
    "path/to/feature2.npy": "class_B",
    "path/to/feature3.npy": "class_A"
}
```
- **Key**: Path to the input feature file (`.npy`).
- **Value**: Predicted class label.

#### **Usage**  
Run the inference script as follows:
```bash
python inference_cls.py <weights_path> <feature_folder> <output_file>
```
- **`weights_path`**: Path to the model weights file.
- **`feature_folder`**: Directory containing input feature files (`.npy`).
- **`output_file`**: Path to save the classification results (JSON format).

## **Feature Input Requirements**  
- The model expects **single, fixed-length vectors** (not time-series or sequential data).
- Each feature file should represent one musical instance (e.g., one song or one description).
- For instance, if the features are 768-dimensional, the expected shape is `(1, 768)` or `(768)`.  
- **All feature files must maintain the same dimensionality.**

## **Naming Convention**  
Ensure all `.npy` feature files follow this naming convention:
```
label_filename.npy
```
> **Note:** The filename should **not** contain underscores (`_`) apart from the one separating the label and the filename.