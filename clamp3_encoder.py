"""
CLAMP3 Unofficial Simplified Interface

This implementation provides a streamlined cross-modal encoding interface designed to reduce the complexity of interacting with the original codebase.
Developers can directly input audio/text paths or content for feature extraction.

Important Notes:
1. Unofficial Implementation: This code differs from the official implementation and is primarily intended for rapid prototyping
2. Testing Limitations: Current version lacks extensive robustness testing. Verify outputs in critical scenarios
3. Edge Case Handling: Treatment of extreme-length inputs and malformed data may deviate from standard implementation

Author: TuTeng
Contact: tuteng0915@gmail.com

Usage Example:
>>> encoder = CLAMP3Encoder(get_global=True)
>>> audio_emb = encoder('audio', "path/to/audio")
>>> print("Audio embedding shape:", audio_emb.shape)

>>> # Text encoding (supports raw text/text file path)
>>> text_emb = encoder("text", "text_or_path/to/text")
>>> print("Text embedding shape:", text_emb.shape)
"""


import os
import numpy as np
import torch
from typing import Union, List
from transformers import BertConfig, AutoTokenizer
import sys
import os
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)
sys.path.extend([
    project_root,                           
    os.path.join(project_root, "code"),     
    os.path.join(project_root, "preprocessing/audio")
])

from code.config import *
from code.utils import *
from hf_pretrains import HuBERTFeature
from extract_mert import mert_infr_features


CLAMP_MODEL_PATH = "./code/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"

class CLAMP3Encoder:
    def __init__(
        self,
        model_path: str = CLAMP_MODEL_PATH,
        device: str = 'cuda',
        text_model_name: str = TEXT_MODEL_NAME,
        max_length: int = 128,
        get_global: bool = True
    ):
        self.model_path = model_path
        self.device = device
        self.initialized = False
        self.get_global = get_global
        
        self.target_sr = 24000
        self.text_model_name = text_model_name
        self.max_length = max_length
        
        self.tokenizer = None
        self.patchilizer = None
        self.model = None
        self.hubert = None

    def _check_initialized(self):
        if not self.initialized:
            self.load_model()
        if not self.initialized:
            raise RuntimeError("Encoder not initialized. Call load_model() first.")

    def load_model(self):
        if self.initialized:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.patchilizer = M3Patchilizer()
        
        symbolic_config = BertConfig(
            vocab_size=1,
            hidden_size=M3_HIDDEN_SIZE,
            num_hidden_layers=PATCH_NUM_LAYERS,
            num_attention_heads=M3_HIDDEN_SIZE//64,
            intermediate_size=M3_HIDDEN_SIZE*4,
            max_position_embeddings=PATCH_LENGTH
        )
        
        audio_config = BertConfig(
            vocab_size=1,
            hidden_size=AUDIO_HIDDEN_SIZE,
            num_hidden_layers=AUDIO_NUM_LAYERS,
            num_attention_heads=AUDIO_HIDDEN_SIZE//64,
            intermediate_size=AUDIO_HIDDEN_SIZE*4,
            max_position_embeddings=MAX_AUDIO_LENGTH
        )
        
        self.model = CLaMP3Model(
            audio_config=audio_config,
            symbolic_config=symbolic_config,
            text_model_name=self.text_model_name,
            hidden_size=768,
            load_m3=False
        ).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint['model'])
        
        self.hubert = HuBERTFeature(
            "m-a-p/MERT-v1-95M",
            24000,
            force_half=False,
            processor_normalize=True,
        ).to(self.device).eval()

        self.model.eval()
        self.initialized = True

    def get_modality_support(self):
        return {'audio': True, 'text': True}

    def encode_text(self, text_input: str) -> torch.Tensor:
        self._check_initialized()
        
        if os.path.isfile(text_input):
            with open(text_input, 'r', encoding='utf-8') as f:
                text = '\n'.join(list(set(f.read().splitlines())))
        else:
            text = text_input
    
        input_data = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_data = input_data['input_ids'].squeeze(0)
        max_input_length = MAX_TEXT_LENGTH
            
        with torch.no_grad():
            return self._encoder_core(input_data, max_input_length, modality="text").unsqueeze(0).cpu().detach().numpy()
        
    def encode_audio(self, audio_path: str) -> torch.Tensor:
        self._check_initialized()
        
        with torch.no_grad():
            wav = mert_infr_features(self.hubert, audio_path, self.device)
            wav = wav.mean(dim=0, keepdim=True)
            input_data = wav.reshape(-1, wav.size(-1)).to(self.device)
            zero_vec = torch.zeros((1, input_data.size(-1))).to(self.device)
            input_data = torch.cat((zero_vec, input_data, zero_vec), 0)
            max_input_length = MAX_AUDIO_LENGTH
            return self._encoder_core(input_data, max_input_length, modality="audio").unsqueeze(0).cpu().detach().numpy()

    def encode_symbolic(self, symbolic_path: str) -> torch.Tensor:
        self._check_initialized()
        
        with open(symbolic_path, 'r', encoding='utf-8') as f:
                symbolic = '\n'.join(list(set(f.read().splitlines())))
        input_data = self.patchilizer.encode(symbolic, add_special_patches=True)
        input_data = torch.tensor(input_data)
        max_input_length = PATCH_LENGTH

        with torch.no_grad():
            return self._encoder_core(input_data, max_input_length, modality="symbolic").unsqueeze(0).cpu().detach().numpy()

    def __call__(self, modality: str, input_data):
        if modality == 'audio':
            return self.encode_audio(input_data)
        elif modality == 'text':
            return self.encode_text(input_data)
        elif modality == 'symbolic':
            return self.encode_text(input_data)
        raise ValueError(f"Unsupported modality: {modality}")

    def _encoder_core(self, input_data, max_input_length, modality="text"):
        segment_list = []
        for i in range(0, len(input_data), max_input_length):
            segment_list.append(input_data[i:i+max_input_length])
        segment_list[-1] = input_data[-max_input_length:]

        last_hidden_states_list = []

        for input_segment in segment_list:
            input_masks = torch.tensor([1]*input_segment.size(0))
            if modality=="text":
                pad_indices = torch.ones(MAX_TEXT_LENGTH - input_segment.size(0)).long() * self.tokenizer.pad_token_id
            elif modality=="symbolic":
                pad_indices = torch.ones((PATCH_LENGTH - input_segment.size(0), PATCH_SIZE)).long() * self.patchilizer.pad_token_id
            else:
                pad_indices = torch.ones((MAX_AUDIO_LENGTH - input_segment.size(0), AUDIO_HIDDEN_SIZE)).float() * 0.
            pad_indices = pad_indices.to(self.device)
            input_masks = torch.cat((input_masks, torch.zeros(max_input_length - input_segment.size(0))), 0).to(self.device)
            input_segment = torch.cat((input_segment, pad_indices), 0).to(self.device)

            if modality=="text":
                last_hidden_states = self.model.get_text_features(text_inputs=input_segment.unsqueeze(0).to(self.device),
                                                            text_masks=input_masks.unsqueeze(0).to(self.device),
                                                            get_global=self.get_global)
            elif modality=="symbolic":
                last_hidden_states = self.model.get_symbolic_features(symbolic_inputs=input_segment.unsqueeze(0).to(self.device),
                                                          symbolic_masks=input_masks.unsqueeze(0).to(self.device),
                                                          get_global=self.get_global)
            else:
                last_hidden_states = self.model.get_audio_features(audio_inputs=input_segment.unsqueeze(0).to(self.device),
                                                            audio_masks=input_masks.unsqueeze(0).to(self.device),
                                                            get_global=self.get_global)
            last_hidden_states_list.append(last_hidden_states)

        if not self.get_global:
            last_hidden_states = last_hidden_states[:, :input_masks.sum().long().item(), :]
        last_hidden_states_list.append(last_hidden_states)

        if not self.get_global:
            last_hidden_states_list = [last_hidden_states[0] for last_hidden_states in last_hidden_states_list]
            last_hidden_states_list[-1] = last_hidden_states_list[-1][-(len(input_data)%max_input_length):]
            last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
        else:
            full_chunk_cnt = len(input_data) // max_input_length
            remain_chunk_len = len(input_data) % max_input_length
            if remain_chunk_len == 0:
                feature_weights = torch.tensor([max_input_length] * full_chunk_cnt, device=self.device).view(-1, 1)
            else:
                feature_weights = torch.tensor([max_input_length] * full_chunk_cnt + [remain_chunk_len], device=self.device).view(-1, 1)
            
            last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
            last_hidden_states_list = last_hidden_states_list * feature_weights
            last_hidden_states_list = last_hidden_states_list.sum(dim=0) / feature_weights.sum()

        return last_hidden_states_list


if  __name__ == "__main__":
    encoder = CLAMP3Encoder()

    audio_emb = encoder('audio', '/data/tteng/MuLM/SMD/1min_audio/0aF9m87P8Tja3NUMv4DfHt.mp3')
    print("audio emb shape:",audio_emb.shape)

    text_emb = encoder("text", "classical piano piece")
    print("text emb shape:",text_emb.shape)