import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from samplings import *
import argparse

class LinearClassification(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassification, self).__init__()
        self.fc = torch.nn.Linear(input_size, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.squeeze()
        x = x.unsqueeze(0)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
def list_files_in_directory(directories, extensions=["npy"]):
    file_list = []
    
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)

    return file_list

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Feature extraction and classification.")
    parser.add_argument("weights_path", type=str, help="Path to the model weights file.")
    parser.add_argument("feature_folder", type=str, help="Directory containing input feature files.")
    parser.add_argument("output_file", type=str, help="File to save the classification results. (format: json)")

    # Parse arguments
    args = parser.parse_args()
    weights_path = args.weights_path
    feature_folder = args.feature_folder
    output_file = args.output_file

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoint = torch.load(weights_path, map_location='cpu')
    print(f"Successfully Loaded Checkpoint from Epoch {checkpoint['epoch']} with acc {checkpoint['max_eval_acc']}")
    input_size = checkpoint['input_size']
    label2idx = checkpoint['labels']
    idx2label = {idx: label for label, idx in label2idx.items()}  # Create reverse mapping
    model = LinearClassification(input_size=input_size, num_classes=len(label2idx))
    model = model.to(device)

    # print parameter number
    print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    model.eval()
    model.load_state_dict(checkpoint['model'])

    # load filenames under train and eval folder
    feature_files = list_files_in_directory([feature_folder])
    cls_results = {}

    for filepath in tqdm(feature_files):
        outputs = np.load(filepath)[0]
        outputs = torch.from_numpy(outputs).to(device)
        outputs = outputs.unsqueeze(0)
        cls_list = model(outputs)[0].tolist()
        max_prob = max(cls_list)
        cls_idx = cls_list.index(max_prob)
        cls_label = idx2label[cls_idx]
        cls_results[filepath] = cls_label

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cls_results, f)
