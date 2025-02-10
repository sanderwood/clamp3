import os
import gc
import time
import wandb
import torch
import random
import weakref
import numpy as np
from utils import *
from config import *
from tqdm import tqdm
from copy import deepcopy
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertConfig, GPT2Config, get_constant_schedule_with_warmup

patchilizer = M3Patchilizer()

def clear_unused_tensors():
    gc.disable()  # Temporarily disable garbage collection
    try:
        # Get the set of tensor ids used by the model
        if hasattr(model, "module"):
            model_tensors = {id(p) for p in model.module.parameters()}
        else:
            model_tensors = {id(p) for p in model.parameters()}
        
        # Get the set of tensor ids used by the optimizer
        optimizer_tensors = {
            id(state) 
            for state_dict in optimizer.state.values() 
            for state in state_dict.values()
            if isinstance(state, torch.Tensor)  # Ensure only tensors are considered
        }

        # List of all CUDA tensors currently in memory
        tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]
        
        # Create weak references to avoid interfering with garbage collection
        tensor_refs = [weakref.ref(tensor) for tensor in tensors]

        for tensor_ref in tensor_refs:
            tensor = tensor_ref()  # Dereference the weak reference
            if tensor is not None and id(tensor) not in model_tensors and id(tensor) not in optimizer_tensors:
                # Mark the tensor for deletion
                tensor.detach_()  # Detach from computation graph
                del tensor  # Delete the tensor reference
    except:
        pass

    finally:
        gc.enable()  # Re-enable garbage collection
        gc.collect()  # Force a garbage collection
        torch.cuda.empty_cache()  # Clear the CUDA cache

def list_files_in_directory(directories, extensions=["abc", "mtf"]):
    file_list = []
    
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)

    return file_list

def collate_batch(batch):
    input_patches, input_masks, selected_indices, target_patches = zip(*batch)

    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=patchilizer.pad_token_id)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True, padding_value=0)
    selected_indices = torch.nn.utils.rnn.pad_sequence(selected_indices, batch_first=True, padding_value=0)
    target_patches = torch.nn.utils.rnn.pad_sequence(target_patches, batch_first=True, padding_value=patchilizer.pad_token_id)

    return input_patches, input_masks, selected_indices, target_patches

class M3Dataset(Dataset):
    def __init__(self, filenames, mode):
        print("The number of "+mode+" data: "+str(len(filenames)))
        self.filenames = filenames
        self.mode = mode
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        try:
            with open(filename, "r", encoding="utf-8") as f:
                item = f.read()
        except Exception as e:
            print(e)
            print("Failed to load: "+filename)
            item = ""

        target_patches = patchilizer.encode(item, add_special_patches=True, truncate=True, random_truncate=(self.mode=="train"))
        input_masks = torch.tensor([1]*len(target_patches))
        input_patches, selected_indices = mask_patches(target_patches, patchilizer, self.mode)
        input_patches = input_patches.reshape(-1)
        target_patches = torch.tensor(target_patches).reshape(-1)
        return input_patches, input_masks, selected_indices, target_patches
    
# call model with a batch of input
def process_one_batch(batch):
    input_patches, input_masks, selected_indices, target_patches = batch
    
    loss = model(input_patches, 
                 input_masks, 
                 selected_indices, 
                 target_patches).loss

    # Reduce the loss on GPU 0
    if world_size > 1:
        loss = loss.unsqueeze(0)
        dist.reduce(loss, dst=0)
        loss = loss / world_size
        dist.broadcast(loss, src=0)

    return loss.mean()

# do one epoch for training
def train_epoch(epoch):
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    model.train()
    train_steps = (epoch-1)*len(train_set)

    for batch in tqdm_train_set:
        with autocast(device_type='cuda'):
            loss = process_one_batch(batch)
        scaler.scale(loss).backward()
        total_train_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        tqdm_train_set.set_postfix({str(global_rank)+'_train_loss': total_train_loss / iter_idx})
        train_steps += 1
        
        # Log the training loss to wandb
        if global_rank==0 and M3_WANDB_LOG:
            wandb.log({"train_loss": total_train_loss / iter_idx}, step=train_steps)

        iter_idx += 1
        if iter_idx % 1000 == 0:
            clear_unused_tensors()
        
    return total_train_loss / (iter_idx-1)

# do one epoch for eval
def eval_epoch():
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    iter_idx = 1
    model.eval()
  
    # Evaluate data for one epoch
    for batch in tqdm_eval_set: 
        with torch.no_grad():
            loss = process_one_batch(batch)

        total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({str(global_rank)+'_eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1

    return total_eval_loss / (iter_idx-1)

# train and eval
if __name__ == "__main__":

    # Set up distributed training
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend='nccl')
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
    if M3_DETERMINISTIC:
        seed = 42 + global_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    encoder_config = BertConfig(vocab_size=1,
                                hidden_size=M3_HIDDEN_SIZE,
                                num_hidden_layers=PATCH_NUM_LAYERS,
                                num_attention_heads=M3_HIDDEN_SIZE//64,
                                intermediate_size=M3_HIDDEN_SIZE*4,
                                max_position_embeddings=PATCH_LENGTH)
    decoder_config = GPT2Config(vocab_size=128,
                                n_positions=PATCH_SIZE,
                                n_embd=M3_HIDDEN_SIZE,
                                n_layer=TOKEN_NUM_LAYERS,
                                n_head=M3_HIDDEN_SIZE//64,
                                n_inner=M3_HIDDEN_SIZE*4)
    model = M3Model(encoder_config, decoder_config)
    model = model.to(device)

    # print parameter number
    print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=True)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=M3_LEARNING_RATE)

    if M3_WANDB_LOG and global_rank==0:
        # Initialize wandb
        if WANDB_KEY:
            wandb.login(key=WANDB_KEY)
        wandb.init(project="m3", 
                   name=M3_WEIGHTS_PATH.replace("weights_", "").replace(".pth", ""))
             
    # load filenames under train and eval folder
    train_files = list_files_in_directory(M3_TRAIN_FOLDERS)
    eval_files = list_files_in_directory(M3_EVAL_FOLDERS)

    if len(eval_files)==0:
        train_files, eval_files = split_data(train_files)
       
    train_batch_nums = int(len(train_files) / M3_BATCH_SIZE)
    eval_batch_nums = int(len(eval_files) / M3_BATCH_SIZE)

    train_files = train_files[:train_batch_nums*M3_BATCH_SIZE]
    eval_files = eval_files[:eval_batch_nums*M3_BATCH_SIZE]

    train_set = M3Dataset(train_files, 'train')
    eval_set = M3Dataset(eval_files, 'eval')

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=global_rank)
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=global_rank)
    
    train_set = DataLoader(train_set, batch_size=M3_BATCH_SIZE, collate_fn=collate_batch, sampler=train_sampler, shuffle = (train_sampler is None))
    eval_set = DataLoader(eval_set, batch_size=M3_BATCH_SIZE, collate_fn=collate_batch, sampler=eval_sampler, shuffle = (train_sampler is None))

    lr_scheduler = get_constant_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = 1000)

    if M3_LOAD_CKPT and os.path.exists(M3_WEIGHTS_PATH):
        # Load checkpoint to CPU
        checkpoint = torch.load(M3_WEIGHTS_PATH, map_location='cpu', weights_only=True)

        # Here, model is assumed to be on GPU
        # Load state dict to CPU model first, then move the model to GPU
        if torch.cuda.device_count() > 1:
            # If you have a DataParallel model, you need to load to model.module instead
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            # Load to a CPU clone of the model, then load back
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_sched'])
        pre_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        min_eval_loss = checkpoint['min_eval_loss']
        print(f"Successfully Loaded Checkpoint from Epoch {checkpoint['epoch']} with loss {checkpoint['min_eval_loss']}")
        checkpoint = None
    
    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = float('inf')

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=M3_LEARNING_RATE)

    for epoch in range(1+pre_epoch, M3_NUM_EPOCH+1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)
        train_loss = train_epoch(epoch)
        eval_loss = eval_epoch()
        if global_rank==0:
            with open(M3_LOGS_PATH,'a') as f:
                f.write("Epoch " + str(epoch) + "\ntrain_loss: " + str(train_loss) + "\neval_loss: " +str(eval_loss) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")
            if eval_loss < min_eval_loss:
                best_epoch = epoch
                min_eval_loss = eval_loss
                checkpoint = { 
                                'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_sched': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                'min_eval_loss': min_eval_loss
                                }
                torch.save(checkpoint, M3_WEIGHTS_PATH)

        if world_size > 1:
            dist.barrier()

    if global_rank==0:
        print("Best Eval Epoch : "+str(best_epoch))
        print("Min Eval Loss : "+str(min_eval_loss))
