import re
import os
import math
import torch
import random
from config import *
from unidecode import unidecode
from torch.nn import functional as F
from transformers import AutoModel, BertModel, GPT2LMHeadModel, PreTrainedModel, GPT2Config

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

class ClipLoss(torch.nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def gather_features(
            self,
            image_features,
            text_features,
            local_loss=False,
            gather_with_grad=False,
            rank=0,
            world_size=1,
            use_horovod=False
    ):
        assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
        if use_horovod:
            assert hvd is not None, 'Please install horovod'
            if gather_with_grad:
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            else:
                with torch.no_grad():
                    all_image_features = hvd.allgather(image_features)
                    all_text_features = hvd.allgather(text_features)
                if not local_loss:
                    # ensure grads for local rank when all_* features don't have a gradient
                    gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                    gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                    gathered_image_features[rank] = image_features
                    gathered_text_features[rank] = text_features
                    all_image_features = torch.cat(gathered_image_features, dim=0)
                    all_text_features = torch.cat(gathered_text_features, dim=0)
        else:
            # We gather tensors from all gpus
            if gather_with_grad:
                all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
                all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            else:
                gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
                gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
                dist.all_gather(gathered_image_features, image_features)
                dist.all_gather(gathered_text_features, text_features)
                if not local_loss:
                    # ensure grads for local rank when all_* features don't have a gradient
                    gathered_image_features[rank] = image_features
                    gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)

        return all_image_features, all_text_features

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = self.gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    
class M3Patchilizer:
    def __init__(self):
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3

    def split_bars(self, body):
        bars = re.split(self.regexPattern, ''.join(body))
        bars = list(filter(None, bars))  # remove empty strings
        if bars[0] in self.delimiters:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        bars = [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]
        return bars
    
    def bar2patch(self, bar, patch_size=PATCH_SIZE):
        patch = [self.bos_token_id] + [ord(c) for c in bar] + [self.eos_token_id]
        patch = patch[:patch_size]
        patch += [self.pad_token_id] * (patch_size - len(patch))
        return patch
    
    def patch2bar(self, patch):
        return ''.join(chr(idx) if idx > self.mask_token_id else '' for idx in patch)

    def encode(self,
               item,
               patch_size=PATCH_SIZE,
               add_special_patches=False,
               truncate=False,
               random_truncate=False):
        item = item.replace("L:1/8\n", "")
        item = unidecode(item)
        lines = re.findall(r'.*?\n|.*$', item)
        lines = list(filter(None, lines))  # remove empty lines

        patches = []

        if lines[0].split(" ")[0] == "ticks_per_beat":
            patch = ""
            for line in lines:
                if patch.startswith(line.split(" ")[0]) and (len(patch) + len(" ".join(line.split(" ")[1:])) <= patch_size-2):
                    patch = patch[:-1] + "\t" + " ".join(line.split(" ")[1:])
                else:
                    if patch:
                        patches.append(patch)
                    patch = line
            if patch!="":
                patches.append(patch)
        else:
            for line in lines:
                if len(line) > 1 and ((line[0].isalpha() and line[1] == ':') or line.startswith('%%')):
                    patches.append(line)
                else:
                    bars = self.split_bars(line)
                    if bars:
                        bars[-1] += '\n'
                        patches.extend(bars)
        
        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * patch_size
            eos_patch = chr(self.eos_token_id) * patch_size
            patches = [bos_patch] + patches + [eos_patch]

        if len(patches) > PATCH_LENGTH and truncate:
            choices = ["head", "tail", "middle"]
            choice = random.choice(choices)
            if choice=="head" or random_truncate==False:
                patches = patches[:PATCH_LENGTH]
            elif choice=="tail":
                patches = patches[-PATCH_LENGTH:]
            else:
                start = random.randint(1, len(patches)-PATCH_LENGTH)
                patches = patches[start:start+PATCH_LENGTH]

        patches = [self.bar2patch(patch) for patch in patches]

        return patches

    def decode(self, patches):
        return ''.join(self.patch2bar(patch) for patch in patches)

class M3PatchEncoder(PreTrainedModel):
    def __init__(self, config):
        super(M3PatchEncoder, self).__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_SIZE*128, M3_HIDDEN_SIZE)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = BertModel(config=config)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3
        
    def forward(self,
                input_patches, # [batch_size, seq_length, hidden_size]
                input_masks):  # [batch_size, seq_length]
        # Transform input_patches into embeddings
        input_patches = torch.nn.functional.one_hot(input_patches, num_classes=128)
        input_patches = input_patches.reshape(len(input_patches), -1, PATCH_SIZE*128).type(torch.FloatTensor)
        input_patches = self.patch_embedding(input_patches.to(self.device))

        # Apply BERT model to input_patches and input_masks
        return self.base(inputs_embeds=input_patches, attention_mask=input_masks)

class M3TokenDecoder(PreTrainedModel):    
    def __init__(self, config):
        super(M3TokenDecoder, self).__init__(config)
        self.base = GPT2LMHeadModel(config=config)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3
        
    def forward(self,
                patch_features,  # [batch_size, hidden_size]
                target_patches): # [batch_size, seq_length]
        # get input embeddings
        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)

        # concatenate the encoded patches with the input embeddings
        inputs_embeds = torch.cat((patch_features.unsqueeze(1), inputs_embeds[:,1:,:]), dim=1)

        # preparing the labels for model training
        target_masks = target_patches == self.pad_token_id
        target_patches = target_patches.clone().masked_fill_(target_masks, -100)

        # get the attention mask
        target_masks = ~target_masks
        target_masks = target_masks.type(torch.int)

        return self.base(inputs_embeds=inputs_embeds,
                         attention_mask=target_masks,
                         labels=target_patches)
        
    def generate(self,
                 patch_feature,
                 tokens):
        # reshape the patch_feature and tokens
        patch_feature = patch_feature.reshape(1, 1, -1)
        tokens = tokens.reshape(1, -1)

        # get input embeddings
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)

        # concatenate the encoded patches with the input embeddings
        tokens = torch.cat((patch_feature, tokens[:,1:,:]), dim=1)
        
        # get the outputs from the model
        outputs = self.base(inputs_embeds=tokens)
        
        # get the probabilities of the next token
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)

        return probs.detach().cpu().numpy()

class M3Model(PreTrainedModel):
    def __init__(self, encoder_config, decoder_config):
        super(M3Model, self).__init__(encoder_config)
        self.encoder = M3PatchEncoder(encoder_config)
        self.decoder = M3TokenDecoder(decoder_config)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3
        
    def forward(self,
                input_patches,      # [batch_size, seq_length, hidden_size]
                input_masks,        # [batch_size, seq_length]
                selected_indices,   # [batch_size, seq_length]
                target_patches):    # [batch_size, seq_length, hidden_size]
        input_patches = input_patches.reshape(len(input_patches), -1, PATCH_SIZE).to(self.device)
        input_masks = input_masks.to(self.device)
        selected_indices = selected_indices.to(self.device)
        target_patches = target_patches.reshape(len(target_patches), -1, PATCH_SIZE).to(self.device)

        # Pass the input_patches and input_masks through the encoder
        outputs = self.encoder(input_patches, input_masks)["last_hidden_state"]
        
        # Use selected_indices to form target_patches
        target_patches = target_patches[selected_indices.bool()]
        patch_features = outputs[selected_indices.bool()]
        
        # Pass patch_features and target_patches through the decoder
        return self.decoder(patch_features, target_patches)

class CLaMP3Model(PreTrainedModel):
    def __init__(self,
                 audio_config,
                 symbolic_config,
                 global_rank=None,
                 world_size=None,
                 text_model_name=TEXT_MODEL_NAME,
                 hidden_size=CLAMP3_HIDDEN_SIZE,
                 load_m3=CLAMP3_LOAD_M3):
        super(CLaMP3Model, self).__init__(symbolic_config)

        self.text_model = AutoModel.from_pretrained(text_model_name) # Load the text model
        self.text_proj = torch.nn.Linear(self.text_model.config.hidden_size, hidden_size) # Linear layer for text projections
        torch.nn.init.normal_(self.text_proj.weight, std=0.02) # Initialize weights with normal distribution

        self.symbolic_model = M3PatchEncoder(symbolic_config) # Initialize the symbolic model
        self.symbolic_proj = torch.nn.Linear(M3_HIDDEN_SIZE, hidden_size) # Linear layer for symbolic projections
        torch.nn.init.normal_(self.symbolic_proj.weight, std=0.02) # Initialize weights with normal distribution

        self.audio_model = BertModel(audio_config) # Initialize the audio model
        self.audio_proj = torch.nn.Linear(audio_config.hidden_size, hidden_size) # Linear layer for audio projections
        torch.nn.init.normal_(self.audio_proj.weight, std=0.02) # Initialize weights with normal distribution

        if global_rank==None or world_size==None:
            global_rank = 0
            world_size = 1

        self.loss_fn = ClipLoss(local_loss=False,
                                gather_with_grad=True,
                                cache_labels=False,
                                rank=global_rank,
                                world_size=world_size,
                                use_horovod=False)

        if load_m3 and os.path.exists(M3_WEIGHTS_PATH):
            checkpoint = torch.load(M3_WEIGHTS_PATH, map_location='cpu', weights_only=True)
            decoder_config = GPT2Config(vocab_size=128,
                            n_positions=PATCH_SIZE,
                            n_embd=M3_HIDDEN_SIZE,
                            n_layer=TOKEN_NUM_LAYERS,
                            n_head=M3_HIDDEN_SIZE//64,
                            n_inner=M3_HIDDEN_SIZE*4)
            model = M3Model(symbolic_config, decoder_config)
            model.load_state_dict(checkpoint['model'])
            self.symbolic_model = model.encoder
            model = None
            print(f"Successfully Loaded M3 Checkpoint from Epoch {checkpoint['epoch']} with loss {checkpoint['min_eval_loss']}")
        
    def set_trainable(self, freeze_list):
        if "text_model" in freeze_list:
            self.text_model.eval()
            for param in self.text_model.parameters():
                param.requires_grad = False
            print("Text Model Frozen")
        else:
            self.text_model.train()
            for param in self.text_model.parameters():
                param.requires_grad = True
            print("Text Model Training")

        if "text_proj" in freeze_list:
            self.text_proj.eval()
            for param in self.text_proj.parameters():
                param.requires_grad = False
            print("Text Projection Layer Frozen")
        else:
            self.text_proj.train()
            for param in self.text_proj.parameters():
                param.requires_grad = True
            print("Text Projection Layer Training")

        if "symbolic_model" in freeze_list:
            self.symbolic_model.eval()
            for param in self.symbolic_model.parameters():
                param.requires_grad = False
            print("Symbolic Model Frozen")
        else:
            self.symbolic_model.train()
            for param in self.symbolic_model.parameters():
                param.requires_grad = True
            print("Symbolic Model Training")

        if "symbolic_proj" in freeze_list:
            self.symbolic_proj.eval()
            for param in self.symbolic_proj.parameters():
                param.requires_grad = False
            print("Symbolic Projection Layer Frozen")
        else:
            self.symbolic_proj.train()
            for param in self.symbolic_proj.parameters():
                param.requires_grad = True
            print("Symbolic Projection Layer Training")

        if "audio_model" in freeze_list:
            self.audio_model.eval()
            for param in self.audio_model.parameters():
                param.requires_grad = False
            print("Audio Model Frozen")
        else:
            self.audio_model.train()
            for param in self.audio_model.parameters():
                param.requires_grad = True
            print("Audio Model Training")

        if "audio_proj" in freeze_list:
            self.audio_proj.eval()
            for param in self.audio_proj.parameters():
                param.requires_grad = False
            print("Audio Projection Layer Frozen")
        else:
            self.audio_proj.train()
            for param in self.audio_proj.parameters():
                param.requires_grad = True
            print("Audio Projection Layer Training")

    def avg_pooling(self, input_features, input_masks):
        input_masks = input_masks.unsqueeze(-1).to(self.device) # add a dimension to match the feature dimension
        input_features = input_features * input_masks # apply mask to input_features
        avg_pool = input_features.sum(dim=1) / input_masks.sum(dim=1) # calculate average pooling
        
        return avg_pool
    
    def get_text_features(self,
                          text_inputs,
                          text_masks,
                          get_global=False):
        text_features = self.text_model(text_inputs.to(self.device),
                                        attention_mask=text_masks.to(self.device))['last_hidden_state']

        if get_global:
            text_features = self.avg_pooling(text_features, text_masks)
            text_features = self.text_proj(text_features)
        
        return text_features
    
    def get_symbolic_features(self,
                              symbolic_inputs,
                              symbolic_masks,
                              get_global=False):
        symbolic_features = self.symbolic_model(symbolic_inputs.to(self.device),
                                                symbolic_masks.to(self.device))['last_hidden_state']

        if get_global:
            symbolic_features = self.avg_pooling(symbolic_features, symbolic_masks)
            symbolic_features = self.symbolic_proj(symbolic_features)
        
        return symbolic_features
    
    def get_audio_features(self,
                           audio_inputs,
                           audio_masks,
                           get_global=False):
        audio_features = self.audio_model(inputs_embeds=audio_inputs.to(self.device),
                                          attention_mask=audio_masks.to(self.device))['last_hidden_state']

        if get_global:
            audio_features = self.avg_pooling(audio_features, audio_masks)
            audio_features = self.audio_proj(audio_features)
        
        return audio_features

    def forward(self,
                text_inputs,     # [batch_size, seq_length]
                text_masks,      # [batch_size, seq_length]
                music_inputs,    # [batch_size, seq_length, hidden_size]
                music_masks,     # [batch_size, seq_length]
                music_modality): # "symbolic" or "audio"
        # Compute the text features
        text_features = self.get_text_features(text_inputs, text_masks, get_global=True)

        # Compute the music features
        if music_modality=="symbolic":
            music_features = self.get_symbolic_features(music_inputs, music_masks, get_global=True)
        elif music_modality=="audio":
            music_features = self.get_audio_features(music_inputs, music_masks, get_global=True)
        else:
            raise ValueError("music_modality must be either 'symbolic' or 'audio'")

        return self.loss_fn(text_features,
                            music_features,
                            LOGIT_SCALE,
                            output_dict=False)

def split_data(data, eval_ratio=EVAL_SPLIT):
    random.shuffle(data)
    split_idx = int(len(data)*eval_ratio)
    eval_set = data[:split_idx]
    train_set = data[split_idx:]
    return train_set, eval_set

def mask_patches(target_patches, patchilizer, mode):
    indices = list(range(len(target_patches)))
    random.shuffle(indices)
    selected_indices = indices[:math.ceil(M3_MASK_RATIO*len(indices))]
    sorted_indices = sorted(selected_indices)
    input_patches = torch.tensor(target_patches)

    if mode=="eval":
        choice = "original"
    else:
        choice = random.choices(["mask", "shuffle", "original"], weights=[0.8, 0.1, 0.1])[0]
    
    if choice=="mask":
        input_patches[sorted_indices] = torch.tensor([patchilizer.mask_token_id]*PATCH_SIZE)
    elif choice=="shuffle":
        for idx in sorted_indices:
            patch = input_patches[idx]
            try:
                index_eos = (patch == patchilizer.eos_token_id).nonzero().item()
            except:
                index_eos = len(patch)

            indices = list(range(1, index_eos))
            random.shuffle(indices)
            indices = [0] + indices + list(range(index_eos, len(patch)))
            input_patches[idx] = patch[indices]
    
    selected_indices = torch.zeros(len(target_patches))
    selected_indices[sorted_indices] = 1.

    return input_patches, selected_indices
    
def remove_instrument_info(item):
    # remove instrument information from symbolic music
    lines = re.findall(r'.*?\n|.*$', item)
    lines = list(filter(None, lines))
    if lines[0].split(" ")[0] == "ticks_per_beat":
        type = "mtf"
    else:
        type = "abc"

    cleaned_lines = []
    for line in lines:
        if type=="abc" and line.startswith("V:"):
            # find the position of " nm=" or " snm="
            nm_pos = line.find(" nm=")
            snm_pos = line.find(" snm=")
            # keep the part before " nm=" or " snm="
            if nm_pos != -1:
                line = line[:nm_pos]
            elif snm_pos != -1:
                line = line[:snm_pos]
            if nm_pos != -1 or snm_pos != -1:
                line += "\n"
        elif type=="mtf" and line.startswith("program_change"):
            line = " ".join(line.split(" ")[:-1]) + " 0\n"
                
        cleaned_lines.append(line)
        
    return ''.join(cleaned_lines)