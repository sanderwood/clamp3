# Music Classification Codebase  

## Overview  
This codebase provides scripts for training and using a linear classification model for music-related feature representations. It supports both training a classifier from extracted features and performing inference on new data.  

## Repository Structure  
The [classification/](https://github.com/sanderwood/clamp3/tree/main/classification) folder contains the following scripts:  

### 1. [inference_cls.py](https://github.com/sanderwood/clamp3/blob/main/classification/inference_cls.py)
This script performs classification on extracted feature vectors using a pre-trained linear probe model.  

#### JSON Output Format  
The output is a JSON file structured as follows:  
```json
{
    "path/to/feature1.npy": "class_A",
    "path/to/feature2.npy": "class_B",
    "path/to/feature3.npy": "class_A"
}
```
- **Key**: Path to the input feature file (`.npy`).  
- **Value**: Predicted class label.  

#### Usage  
```bash
python inference_cls.py <weights_path> <feature_folder> <output_file>
```
- `weights_path`: Path to the model weights file.  
- `feature_folder`: Directory containing input feature files (`.npy`).  
- `output_file`: Path to save the classification results (JSON format).  

### 2. [train_cls.py](https://github.com/sanderwood/clamp3/blob/main/classification/train_cls.py)
This script trains a linear classification model on extracted features.  

#### Arguments  
- `--train_folder`: Path to the training folder.  
- `--eval_folder`: Path to the evaluation folder. If not specified, 20% of the training data will be randomly set aside for evaluation.  
- `--eval_split`: Fraction of training data used for evaluation (default: 0.2).  
- `--wandb_key`: Weights & Biases API key for logging.  
- `--num_epochs`: Maximum number of training epochs (default: 1000).  
- `--learning_rate`: Learning rate for optimization (default: 1e-5).  
- `--balanced_training`: Flag to balance labels in training data.  
- `--wandb_log`: Flag to log training metrics to Weights & Biases.  

#### Usage  
```bash
python train_cls.py --train_folder <path> --eval_folder <path> [--eval_split <float>] [--num_epochs <int>] [--learning_rate <float>] [--balanced_training] [--wandb_log]
```
Example:
```bash
python train_cls.py --train_folder data/train
```

### Feature Input Requirements  
The model expects **single, fixed-length vectors** and does not support time-series or sequential input. Each feature file should contain one feature vector per musical instance (e.g., one song or one description).

If the features are 768-dimensional, the shape should be `(1, 768)`. All feature files should have the same dimension.

## Naming Convention  
All `.npy` feature files must follow the naming convention `label_filename.npy`, where the filename should not contain underscores (`_`).