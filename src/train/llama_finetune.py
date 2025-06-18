
import argparse
import os
import torch
import transformers

from CAD_dataset import CADDataset, DataCollatorForSupervisedDataset
from huggingface_hub import login
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from utils import prepare_model_and_tokenizer

login() # put your huggingface token here

def setup_datasets(args, llama_tokenizer, transform_args={}):
    datasets = {
        "train": CADDataset(
            args.data_path,
            llama_tokenizer=llama_tokenizer,
        ),
        "val": CADDataset(
            args.eval_data_path,
            llama_tokenizer=llama_tokenizer,
        ),
    }
    return datasets


def setup_training_args(args):
    output_dir = args.expdir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        os.environ["WANDB_DISABLED"] = "True"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    training_args = TrainingArguments(
        fsdp=False,
        fp16=False,
        bf16=False,
        do_eval=True,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        num_train_epochs=args.num_epochs,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=10,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.num_warmup_steps,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=output_dir,
        run_name=args.run_name,
        report_to="wandb",
        dataloader_num_workers=8,
        remove_unused_columns=False,
        # label_names=["cad_ids"],  # this is to make trainer behave as expected
    )
    return training_args


def setup_trainer(args):
    training_args = setup_training_args(args)
    if args.device_map == 'accelerate':
        args.device_map = {'': training_args.local_rank}
    model, llama_tokenizer = prepare_model_and_tokenizer(args)

    datasets = setup_datasets(args, llama_tokenizer)

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=llama_tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=data_collator,
    )

    return trainer


def main(args):
    trainer = setup_trainer(args)

    if args.resume_dir is not None:
        train_result = trainer.train(resume_from_checkpoint=args.resume_dir)
    else:
        train_result = trainer.train()

    print(train_result)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--expdir", type=Path, default="exp")
    parser.add_argument("--model-name", default="llama3")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--data-path", type=Path, default="data/train.json")
    parser.add_argument("--eval-data-path", type=Path, default="data/eval.json")
    parser.add_argument("--pretrained-path", type=Path, default=None)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--num-warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-freq", default=1000, type=int)
    parser.add_argument("--save-freq", default=50000, type=int)
    parser.add_argument("--device-map", type=str, default='auto')
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    os.environ["WANDB_PROJECT"] = "CADFusion_SL"
    main(args)
