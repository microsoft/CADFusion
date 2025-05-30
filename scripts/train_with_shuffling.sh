# set it to your data path
data_path=data/sl_data
# set it to your experiment path
exp_path=exp/model_ckpt
train_data=$data_path/train.json
eval_data=$data_path/val.json
shuffle_dataset_between_x_epochs=2
mkdir -p $exp_path

# round 0
accelerate launch --config_file ds_config.yaml src/train/llama_finetune.py --lora-rank 32 --lora-alpha 32 \
        --num-epochs $shuffle_dataset_between_x_epochs --run-name $1 --data-path $train_data --eval-data-path $eval_data \
        --device-map accelerate --eval-freq 1000 --save-freq 50000 --model-name llama3 --expdir $exp_path

for round in 1 2 3 4 5 6 7 8 9
do
    python src/train/llama_finetune.py --lora-rank 32 --pretrained-path $exp_path/$1 --lora-alpha 32 \
        --num-epochs $shuffle_dataset_between_x_epochs --run-name $1 --data-path $train_data --eval-data-path $eval_data \
        --eval-freq 4000 --save-freq 50000 --expdir $exp_path
done
