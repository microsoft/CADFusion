import argparse
import os
import torch
import json
import random
import transformers
from huggingface_hub import login

login() # put your huggingface token here
os.environ["WANDB_PROJECT"] = "CADFusion_VF"

from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from utils import prepare_model_and_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", type=str, required=True)
parser.add_argument("--lora-rank", type=int, default=32)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--lora-dropout", type=float, default=0.05)
parser.add_argument("--sample-cutoff", default=100000, type=int)
parser.add_argument("--pretrained-path", type=str, required=True)
parser.add_argument("--data-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)
parser.add_argument("--num-epochs", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--eval-freq", default=1000, type=int)
parser.add_argument("--save-freq", default=500, type=int)
parser.add_argument("--debug", action="store_true", default=False)
args = parser.parse_args()



with open(args.data_path, 'r') as f:
    raw_data = json.load(f)
    
random.shuffle(raw_data)

if len(raw_data) > args.sample_cutoff + 100:
    ds = {
        "train": Dataset.from_list(raw_data[:args.sample_cutoff]),
        "val": Dataset.from_list(raw_data[-100:])
    }
else:
    ds = {
        "train": Dataset.from_list(raw_data[:-100]),
        "val": Dataset.from_list(raw_data[-100:])
        }

llama_model, llama_tokenizer = prepare_model_and_tokenizer(args)

for name, param in llama_model.named_parameters():
    if "lora" in name:  # Check if "lora" is in the parameter's name
        param.requires_grad = True
        
training_args = DPOConfig(
    run_name=args.run_name,
    learning_rate=1.41e-5, 
    per_device_train_batch_size=2,
    per_device_eval_batch_size=args.batch_size,
    report_to="wandb",
    num_train_epochs=args.num_epochs,
    do_eval=True,
    eval_steps=args.eval_freq,
    save_steps=args.save_freq,
    output_dir=args.output_path
    )

trainer = DPOTrainer(
    llama_model,
    None,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['val'],
    tokenizer=llama_tokenizer,
)
trainer.save_model()
trainer.train()
trainer.save_model()