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

TODO: add environment requirements.

## Data preparation
We provide the human-annotated text-to-CAD dataset we used for training. If you want to train everything from scratch, please follow the instructions below.

### From Scratch
We provide a script to execute all the preprocessing steps until human annotation. 
```
./scripts/preprocess_skexgen.sh
```
If you want to customize the internal steps, expand the following section for more details.
<details>
<summary>Start from scratch (click to expand).</summary>

1. Download the [SkexGen](https://github.com/samxuxiang/SkexGen) data by: [Google Drive link](https://drive.google.com/file/d/1so_CCGLIhqGEDQxMoiR--A4CQk4MjuOp/view).

```
gdown --id 1so_CCGLIhqGEDQxMoiR--A4CQk4MjuOp
unzip cad_data.zip
```

2. Convert the SkexGen data into sequences. Note that `train_deduplicate_s.pkl`, `val.pkl` and `test.pkl` should be converted separately.
```
python3 src/data_preprocessing/convert.py --in_path <skexgen_path> --out_path <sequence_path>
```

3. Render the sequences into images. *Note that running the last step on linux requires the installation of an x server (e.g. `xvfb`). See [this discussion.](https://github.com/tpaviot/pythonocc-core/issues/1302#issuecomment-2053526444)*
```
python3 src/rendering_utils/parser.py --in-path <sequence_path> --out-path <visual_object_folder>
timeout 180 python3 src/rendering_utils/parser_visual.py --data_folder <visual_object_folder>
python3 src/rendering_utils/img_renderer.py --input_dir <visual_object_folder> --output_dir <image_folder>
```

4. Annotate these data with LLM captioning.
```
# Generic:
python3 src/data_preprocessing/captioning.py --image-folder-path <image_folder> --out-path <sl_data_path>

```
* We use openai and azure system for LLM calling. You are welcome to use your own LLMs and prompts by changing `line 21, 22` of `src/data_preprocessing/captioning.py` with your own client definition and function calls.
</details>

### From Our Preprocessed Data
Our preprocessed and annotated dataset can be found in [TODO: data path](todo). For this repo, download the dataset and place it in `data/sl_data` folder. It should contain the following files:
```
data/sl_data
├── train.json
├── val.json
├── test.json
```

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
## Visual Learning
For an individual round of visual learning, run
```
python src/train/dpo.py --run-name <dpo_run_name> --pretrained-path <pretrained_model_path> --data-path <dpo_data_Path> --output-path <model_saving_path>
```
By default it runs dpo for 3 epochs, but you can change by adding flag `--num-epochs x`.

We provide a script for executing our alternate training round. See `scripts/alternate_VF.sh`.
```
./scripts/alternate_VF.sh  # change the value of base_name in the script as instructed
```

## Inference & Visualization
Use `scripts/generate_samples.sh`.
```
./scripts/generate_samples.sh <run_name> test --full
```

## Evaluation
TBD...