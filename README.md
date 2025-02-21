# CADFusion

This repo is the official implementation of paper [Text-to-CAD Generation Through Infusing Visual Feedback in Large Language Models](https://arxiv.org/abs/2501.19054).

## Installation

- Create a conda environment and install all the dependencies.

```
conda env create -f environments.yaml
```

- After installation, activate the environment with

```
conda activate <env>
```

## Data preparation
TBD...

## Sequential Learning
 - A normal training script on multiple GPUs is provided. Change `num_processes` in `ds_config.yaml` to specify how many GPUs will be used.
```
CUDA_VISIBLE_DEVICES=<gpu_ids> accelerate launch --config_file ds_config.yaml src/train/llama_finetune.py \
    --num-epochs <num_epochs> --run-name <run_name> --data-path <train_data> --eval-data-path <eval_data> \
    --device-map accelerate --model-name llama3 --expdir <model_saving_path>
```

 - In our work we shuffle the dataset per x epochs. To train model with this implementation, use `scripts/train_with_shuffling.sh`.
```
./scripts/train_with_shuffling.sh <run_name>
```
## Sequential Learning
For an individual round of sequential learning, run
```
python src/train/dpo.py --run-name <dpo_run_name> --pretrained-path <pretrained_model_path> --data-path <dpo_data_Path> --output-path <model_saving_path>
```
By default it runs dpo for 3 epochs, but you can change by adding flag `--num-epochs x`.

We provide a script for executing our alternate training round. See `scripts/alternate_VF.sh`.
```
./scripts/alternate_VF.sh # change the value of base_name in the script as instructed
```

## Inference
TBD...

## Visualization
TBD...

## Evaluation
TBD...