# by default set it to CADFusion/data
data_path=/your/path/to/data/folder
# by default set it to CADFusion/exp
exp_path=/your/path/to/exp/folder
# by default set it to CADFusion/data
exp_path=/your/path/to/vf_data/folder
train_data=$data_path/train.json
eval_data=$data_path/eval.json

base_name=model_name_you_trained_for_SL

run_name=${base_name}0
CUDA_VISIBLE_DEVICES=0,1 ./scripts/inference.sh $run_name test "--full --device-map auto" &
CUDA_VISIBLE_DEVICES=2,3 ./scripts/inference.sh $run_name train "--sample-len 1000 --device-map auto"
wait

./scripts/make_dpo_data.sh $run_name --score-only &
./scripts/make_dpo_data.sh $run_name-train "--gpu 1"
wait


for LOOP in 1 2 3 4 5
do
    run_name=$base_name$LOOP    
    dpo_training_path=$vf_path/$base_name$((LOOP-1))-train.json
    dpo_run_name=$base_name$LOOP-dpo
    dpo_save_path=$exp_path/$dpo_run_name
    sft_run_name=$base_name$LOOP

    python src/train/dpo.py --run-name $dpo_run_name --pretrained-path $exp_path/$base_name$((LOOP-1)) --data-path $dpo_training_path --output-path $dpo_save_path
    python src/train/llama_finetune.py --num-epochs 1 --run-name $sft_run_name --data-path $train_data --eval-data-path $eval_data --eval-freq 3000 --pretrained-path $dpo_save_path --expdir $exp_path
    
    CUDA_VISIBLE_DEVICES=0 ./scripts/inference.sh $dpo_run_name test "--full --device-map auto" &
    CUDA_VISIBLE_DEVICES=1 ./scripts/inference.sh $run_name test "--full --device-map auto" &
    CUDA_VISIBLE_DEVICES=2,3 ./scripts/inference.sh $run_name train "--sample-len 1000 --device-map auto"
    wait

    ./scripts/make_dpo_data.sh $dpo_run_name --score-only &
    ./scripts/make_dpo_data.sh $run_name "--score-only --gpu 1" &
    ./scripts/make_dpo_data.sh $run_name-train "--gpu 2"
    wait
done