# CADFusion


This repo is the official implementation of paper **[ICML 2025] Text-to-CAD Generation Through Infusing Visual Feedback in Large Language Models** by *Ruiyu Wang, Yu Yuan, Shizhao Sun, Jiang Bian*.

[Paper](https://arxiv.org/abs/2501.19054) | [Video](https://www.youtube-nocookie.com/embed/LK8LAzR0v5M?si=FD1Vg9wjkROTKjDV) | [Huggingface](https://huggingface.co/microsoft/CADFusion)

CADFusion is a text-to-CAD generation framework that leverages visual feedback to enhance the performance of large language models (LLMs) in generating CAD models from textual descriptions. It consists of two main components: sequential learning and visual learning. The sequential learning component fine-tunes LLMs on a text-to-CAD dataset, while the visual learning component alternates between training a visual feedback model and fine-tuning the LLM with the generated visual feedback.

## Installation

- Create a conda environment and install the generic dependencies.

```
name=<your-env-name>
conda create -n $name python=3.8
conda activate $name
python -m pip install -e .
```

- Install the additional dependencies for training.

```
python -m pip install -e .["train"]
```

- Install the additional dependencies for evaluation and rendering.

```
python -m pip install -e .["render"]
conda install -c conda-forge pythonocc-core=7.7.0
pip install git+https://github.com/otaheri/chamfer_distance@dc9987dcf70888d387d96893ba1fb9ba9a333992
python -m pip install -e .["eval"]
```

## Data Preparation
CADFusion is trained by alternating the **Sequential Learning (SL)** stage and the **Visual Feedback (VF)** stage.
We introduce how to prepare the training data for these two stages in the below.

### Data for Sequential Learning

#### Approach 1: Use Existing Human-Annotated Textual Descriptions
We provide human-annoated textual descriptions and their correspoding CAD model IDs in [Skexgen](https://github.com/samxuxiang/SkexGen) under `data/sl_data/sl_data.zip`. It should contain the following files after unzipping:
```
data/sl_data
├── train.json
├── val.json
├── test.json
```
To use our annotated data, download the SkexGen data, unzip it as the reference dataset and run the convertion script to get the dataset. In detail, run the following command:
```
# make sure you are in the root directory of this repo and have the 'data/sl_data/sl_data.zip' unzipped
gdown --id 1so_CCGLIhqGEDQxMoiR--A4CQk4MjuOp 
unzip cad_data.zip
python3 data/sl_data/convert.py
```
The `train.json`, `val.json` and `test.json` under `data/sl_data` are the datasets.

#### Approach 2: Create Human-Annotated Textual Descriptions by Yourself
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


### Data for Visual Feedback

The VF dataset is generated from automatically from the VF pipeline described in the Visual Learning section. We do not recommend using previously generated VF data, as the policy update should depend on its own generations. We do provide an example of the VF data format [TODO: data path](todo) which you can inspect the formatting by unzipping it and placing it under `data/vf_data/example_vf_data.json`. You can use this as a template for your own VF data if you want to add pairs manually.

## Sequential Learning
We use the following script to train the sequential learning model.
```
./scripts/train_with_shuffling.sh <run_name>
```

You are also welcome to customize the training procedure. A normal training script on multiple GPUs is provided. Change `num_processes` in `ds_config.yaml` to specify how many GPUs will be used.
```
CUDA_VISIBLE_DEVICES=<gpu_ids> accelerate launch --config_file ds_config.yaml src/train/llama_finetune.py \
    --num-epochs <num_epochs> --run-name <run_name> --data-path <train_data> --eval-data-path <eval_data> \
    --device-map accelerate --model-name llama3 --expdir <model_saving_path>
```

In our work we shuffle the dataset per x epochs. To train model with this implementation, inspect and modify `scripts/train_with_shuffling.sh`.

## Visual Learning
We provide a script for executing our alternate training round. See `scripts/alternate_VF.sh`.
```
./scripts/alternate_VF.sh  # change the value of base_name in the script as instructed
```
We also provide a script for training on multiple gpus for saving time: `scripts/alternate_VF.sh`. In our setting, we use 4 GPUs for training. You can change the script to use more GPUs if you have them available.

For an individual round of visual learning, run
```
python src/train/dpo.py --run-name <dpo_run_name> --pretrained-path <pretrained_model_path> --data-path <dpo_data_Path> --output-path <model_saving_path>
```
By default it runs dpo for 3 epochs, but you can change by adding flag `--num-epochs x`.


## Model Checkpoints
We provide two model checkpoints: CADFusion v1.0 and v1.1. You can download them from the following links:
- [TODO: CADFusion v1.0](TODO)
- [TODO: CADFusion v1.1](TODO)
You should unzip them and place them under the `exp/model_ckpt` folder for using.

## Inference & Visualization
Use `scripts/generate_samples.sh`.
```
./scripts/generate_samples.sh <run_name> test --full
```
You can find samples generated in `exp/model_generation/<run_name>.jsonl` and rendered figures under the `exp/figures/<run_name>` folder.

## Evaluation
Use the functions in `src/test`.

## Acknowledgements
We would like to acknowledge that the CAD rendering and distributional metrics in this repository is partially based on and adapted from the [SkexGen](https://github.com/samxuxiang/SkexGen) project.

## Citation
If you find our work useful, please cite the following paper
```
@article{wang2025texttocad,
  title={Text-to-CAD Generation Through Infusing Visual Feedback in Large Language Models},
  author={Wang, Ruiyu and Yuan, Yu and Sun, Shizhao and Bian, Jiang},
  journal={arXiv preprint arXiv:2501.19054},
  year={2025}
}
```
<!-- @inproceedings{wang2025texttocad, 
title     = {Text-to-CAD Generation Through Infusing Visual Feedback in Large Language Models},
author    = {Wang, Ruiyu and Yuan, Yu and Sun, Shizhao and Bian, Jiang},
booktitle = {International Conference on Machine Learning},
pages={??},
year={2025},
organization={PMLR}
} -->