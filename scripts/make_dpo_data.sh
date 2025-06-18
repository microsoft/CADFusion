source_path=exp/model_generation/$1.jsonl
figure_path=exp/figures/$1/
save_path=data/vf_data/$1.json

python src/dpo/make_dpo_dataset.py --source-data-path $source_path --figure-path $figure_path --save-path $save_path --num-samples 5 $2
