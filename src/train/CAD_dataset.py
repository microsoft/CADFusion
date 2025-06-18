import json
import os
import torch
import random
import transformers

from dataclasses import dataclass
from torch.utils.data import Dataset
from utils import IGNORE_INDEX, MAX_LENGTH

class CADDataset(Dataset):
    def __init__(self, json_fn, cutoff=True, llama_tokenizer=None):
        if not os.path.exists(json_fn):
            raise ValueError(f"{json_fn} does not exist")
        self.inputs = json.load(open(json_fn, "r"))
        print(len(self.inputs))
        self.inputs = [item for item in self.inputs if 'null' not in item['description']]
        random.shuffle(self.inputs)
        if cutoff:
            self.inputs = self.inputs[:18953]
        print(len(self.inputs))
        self.llama_tokenizer = llama_tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):  
        item = self.inputs[index]  
        seq = item['command_sequence']  
        des = item['description']
        val = self.tokenize(seq, des)
        return val
    
    
    def tokenize(self, seq, des):  
        tokens, prompt_length = self.conditional_generation_task(seq=seq, des=des)  
        input_ids = tokens.input_ids[0]  
        labels = tokens.input_ids[0].clone()  # Clone the input_ids for labels  
        # Set the labels for the prompt part to IGNORE_INDEX so they are ignored in loss calculation  
        labels[:prompt_length] = IGNORE_INDEX  
        input_id_lens = label_lens = (  
            tokens.input_ids.ne(self.llama_tokenizer.pad_token_id).sum().item()  
        )  
        return dict(  
            input_ids=input_ids,  
            input_id_lens=input_id_lens,  
            labels=labels,  
            label_lens=label_lens,  
        )  

    
    def conditional_generation_task(self, seq, des):  
        prompt = 'Below is a description of a 3D shape:\n'  
        prompt += des  
        prompt += '\nGenerate a Computer-Aided Design(CAD) command sequence of the 3D shape:\n'  
        full_text = prompt + seq + self.llama_tokenizer.eos_token  
        tokens = self.llama_tokenizer(  
            full_text,  
            max_length=MAX_LENGTH,  
            return_tensors="pt",  
            truncation=True,  
        )  
        prompt_length = len(self.llama_tokenizer(prompt)['input_ids'])  
        return tokens, prompt_length  
    
    
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key].clone().detach() for instance in instances]
            for key in ("input_ids", "labels")
        )
        # force left padding
        reversed_sequences = [torch.flip(input_id, [0]) for input_id in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(reversed_sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_ids = torch.flip(input_ids, [0, 1])
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )