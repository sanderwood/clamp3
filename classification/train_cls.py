import os
import time
import math
import wandb
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import get_constant_schedule_with_warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Create an argument parser
parser = argparse.ArgumentParser(description='Configuration for generative modelling and classification')

# Add command-line arguments with default values
parser.add_argument('--train_folder', type=str, default=None,
                    help='Path to the training folder')
parser.add_argument('--eval_folder', type=str, default=None,
                    help='Path to the evaluation folder')
parser.add_argument('--eval_split', type=float, default=0.2,
                    help='Fraction of training data to use for evaluation')
parser.add_argument('--wandb_key', type=str, default="<your_wandb_key>",
                    help='Weights and Biases API key')
parser.add_argument('--num_epochs', type=int, default=1000,
                    help='Max number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='Optimizer learning rate')
parser.add_argument('--balanced_training', action='store_true', default=False,
                    help='Set to True to balance labels in training data')
parser.add_argument('--wandb_log', action='store_true', default=False,
                    help='Set to True to log training metrics to WANDB')

# Parse the command-line arguments
args = parser.parse_args()

# Unpack the parsed arguments into variables
TRAIN_FOLDER = args.train_folder
EVAL_FOLDER = args.eval_folder
EVAL_SPLIT = args.eval_split
WANDB_KEY = args.wandb_key
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
BALANCED_TRAINING = args.balanced_training
WANDB_LOG = args.wandb_log

# Print the configuration parameters
print(f"TRAIN_FOLDER: {TRAIN_FOLDER}")
print(f"EVAL_FOLDER: {EVAL_FOLDER}")
print(f"EVAL_SPLIT: {EVAL_SPLIT}")
print(f"WANDB_KEY: {WANDB_KEY}")
print(f"NUM_EPOCHS: {NUM_EPOCHS}")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"BALANCED_TRAINING: {BALANCED_TRAINING}")
print(f"WANDB_LOG: {WANDB_LOG}")

# Paths Configuration
folder_name = TRAIN_FOLDER.split('/')[-1]
WEIGHTS_PATH = f"weights-{folder_name}.pth"  # Weights file path
LOGS_PATH = f"logs-{folder_name}.txt"  # Log file path

# Set up distributed training
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

if world_size > 1:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl') if world_size > 1 else None
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
# Set random seed
seed = 42 + global_rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = 1

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
    
def collate_batch(input_tensors):

    input_tensors, labels = zip(*input_tensors)
    input_tensors = torch.stack(input_tensors, dim=0)
    labels = torch.stack(labels, dim=0)

    return input_tensors.to(device), labels.to(device)

def list_files_in_directory(directory):
    file_list = []
    if directory and os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".npy"):
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)
    return file_list

class TensorDataset(Dataset):
    def __init__(self, filenames):
        print(f"Loading {len(filenames)} files for classification")
        self.filenames = []
        self.label2idx = {}
        self.hiddensize = 0

        for filename in tqdm(filenames):
            label = os.path.basename(filename).split('_')[0]

            self.filenames.append(filename)
            if label not in self.label2idx:
                self.label2idx[label] = len(self.label2idx)
        data = np.load(self.filenames[0])
        data = data.squeeze()
        self.hiddensize = data.shape[0]
        print(f"Found {len(self.label2idx)} classes")
        print(f"Hidden size: {self.hiddensize}")
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        filename = self.filenames[idx]
        label = os.path.basename(filename).split('_')[0]
        label = self.label2idx[label]
        
        # load numpy file
        data = np.load(filename)
        data = torch.from_numpy(data)
        label = torch.tensor(label)

        return data, label

class BalancedTensorDataset(Dataset):
    def __init__(self, filenames):
        print(f"Loading {len(filenames)} files for classification")
        self.filenames = filenames
        self.label2idx = {}
        self.label2files = {}
        
        for filename in tqdm(filenames):
            label = os.path.basename(filename).split('_')[0]
            if label not in self.label2idx:
                self.label2idx[label] = len(self.label2idx)
            if label not in self.label2files:
                self.label2files[label] = []
            self.label2files[label].append(filename)
        print(f"Found {len(self.label2idx)} classes")
        
        self.min_samples = min(len(files) for files in self.label2files.values())

        self._update_epoch_filenames()

    def _update_epoch_filenames(self):
        self.epoch_filenames = []
        for label, files in self.label2files.items():
            sampled_files = random.sample(files, self.min_samples)
            self.epoch_filenames.extend(sampled_files)

        random.shuffle(self.epoch_filenames)

    def __len__(self):
        return len(self.epoch_filenames)

    def __getitem__(self, idx):
        filename = self.epoch_filenames[idx]
        label = os.path.basename(filename).split('_')[0]
        label = self.label2idx[label]
        
        data = np.load(filename)
        data = torch.from_numpy(data)[0]
        label = torch.tensor(label)

        return data, label

    def on_epoch_end(self):
        self._update_epoch_filenames()

# load filenames under train and eval folder
train_files = list_files_in_directory(TRAIN_FOLDER)
eval_files = list_files_in_directory(EVAL_FOLDER)

if len(eval_files)==0:
    random.shuffle(train_files)
    eval_files = train_files[:math.ceil(len(train_files)*EVAL_SPLIT)]
    train_files = train_files[math.ceil(len(train_files)*EVAL_SPLIT):]
if BALANCED_TRAINING:
    train_set = BalancedTensorDataset(train_files)
else:
    train_set = TensorDataset(train_files)
eval_set = TensorDataset(eval_files)
eval_set.label2idx = train_set.label2idx
input_size = train_set.hiddensize

model = LinearClassification(input_size=input_size,
                             num_classes=len(train_set.label2idx))
model = model.to(device)

# print parameter number
print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=True)

scaler = GradScaler()
is_autocast = True
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# call model with a batch of input
def process_one_batch(batch):
    input_tensors, labels = batch
    logits = model(input_tensors)
    loss = loss_fn(logits, labels)
    prediction = torch.argmax(logits, dim=1)
    acc_num = torch.sum(prediction==labels)

    return loss, acc_num, prediction, labels

# do one epoch for training
def train_epoch():
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    total_acc_num = 0
    iter_idx = 1
    model.train()

    for batch in tqdm_train_set:
        if is_autocast:
            with autocast(device_type='cuda'):
                loss, acc_num, prediction, labels = process_one_batch(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, acc_num, prediction, labels = process_one_batch(batch)
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        total_train_loss += loss.item()
        total_acc_num += acc_num.item()
        tqdm_train_set.set_postfix({str(global_rank)+'_train_acc': total_acc_num / (iter_idx*batch_size)})
        # Log the training loss to wandb
        if global_rank==0 and WANDB_LOG:
            wandb.log({"acc": total_acc_num / (iter_idx*batch_size)})

        iter_idx += 1
    
    if BALANCED_TRAINING:
        train_set.dataset.on_epoch_end()
        
    return total_acc_num / ((iter_idx-1)*batch_size)

# do one epoch for eval
def eval_epoch():
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    total_acc_num = 0
    iter_idx = 1
    model.eval()
  
    all_predictions = []
    all_labels = []
  
    # Evaluate data for one epoch
    for batch in tqdm_eval_set: 
        with torch.no_grad():
            loss, acc_num, prediction, labels = process_one_batch(batch)
            total_eval_loss += loss.item()
            total_acc_num += acc_num.item()

            # Accumulate predictions and labels
            all_predictions.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        tqdm_eval_set.set_postfix({str(global_rank)+'_eval_acc': total_acc_num / (iter_idx*batch_size)})
        iter_idx += 1

    # Compute F1 Macro
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    return total_acc_num / ((iter_idx - 1) * batch_size), f1_macro

# train and eval
if __name__ == "__main__":

    if os.path.exists(WEIGHTS_PATH):
        # end training if weights file already exists
        print("Weights file already exists")
        exit()
        
    label2idx = train_set.label2idx
    max_eval_acc = 0
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=global_rank)
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=global_rank)

    train_set = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_batch, sampler=train_sampler, shuffle = (train_sampler is None))
    eval_set = DataLoader(eval_set, batch_size=batch_size, collate_fn=collate_batch, sampler=eval_sampler, shuffle = (train_sampler is None))

    lr_scheduler = get_constant_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = len(train_set))

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    if WANDB_LOG and global_rank==0:
        # Initialize wandb
        if WANDB_KEY:
            wandb.login(key=WANDB_KEY)
        wandb.init(project="linear", 
                   name=WEIGHTS_PATH.replace("weights_", "").replace(".pth", ""))
             
    for epoch in range(1, NUM_EPOCHS+1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)
        train_acc = train_epoch()
        eval_acc, eval_f1_macro = eval_epoch()
        if global_rank==0:
            with open(LOGS_PATH,'a') as f:
                f.write("Epoch " + str(epoch) + "\ntrain_acc: " + str(train_acc) + "\neval_acc: " +str(eval_acc) + "\neval_f1_macro: " +str(eval_f1_macro) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")
            if eval_acc > max_eval_acc:
                best_epoch = epoch
                max_eval_acc = eval_acc
                checkpoint = { 
                                'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_sched': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                'max_eval_acc': max_eval_acc,
                                "labels": label2idx,
                                "input_size": input_size
                                }
                torch.save(checkpoint, WEIGHTS_PATH)
                with open(LOGS_PATH,'a') as f:
                    f.write("Best Epoch so far!\n\n\n")
        
        if world_size > 1:
            dist.barrier()

    if global_rank==0:
        print("Best Eval Epoch : "+str(best_epoch))
        print("Max Eval Accuracy : "+str(max_eval_acc))
