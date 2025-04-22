source_path=path/to/the/instruction/json
figure_path=path/to/the/rendered/figures
save_path=path/to/save/dpo/data

python src/dpo/make_dpo_dataset.py --source-data-path $source_path --figure-path $figure_path --save-path $save_path --num-samples 5 $2
