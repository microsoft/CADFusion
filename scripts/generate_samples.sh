train_data_path=/path/to/the/training/data
test_data_path=/path/to/the/test/data
run_name=$1
temperature=0.9

if [ -z "$2" ]
  then
    data_path=$test_data_path
else
    if [ $2 = "train" ]; then
        data_path=$train_data_path
        run_name=$1-train
    else
        data_path=$test_data_path
        temperature=0.3
    fi
fi

model_path=/<path/to/the/exp/folder>/exp/$1
inference_path=/<path/to/the/exp/folder>/out/$run_name.jsonl
visual_obj_path=/<path/to/the/exp/folder>/visual_obj/$run_name
output_figure_path=/<path/to/the/exp/folder>/figures/$run_name
log_path=/<path/to/the/exp/folder>/logs/$run_name/

mkdir $log_path

echo "--------------------Inferencing--------------------" > $log_path/inference.txt
rm $inference_path
python3 test/inference.py --pretrained-path $model_path --in-path $data_path --out-path $inference_path --num-samples 5 --temperature $temperature --model-name llama3 > $log_path/inference.txt $3

echo "--------------------Parsing CAD objects--------------------" > $log_path/parsing_cad.txt
rm -rf $visual_obj_path
python3 src/rendering_utils/parser.py --in_path $inference_path --out_path $visual_obj_path > $log_path/parsing_cad.txt

echo "--------------------Parsing visual objects--------------------" > $log_path/parsing_visual.txt
timeout 180 src/rendering_utils/parser_visual.py --data_folder $visual_obj_path > $log_path/parsing_visual.txt

echo "--------------------Rendering--------------------" > $log_path/rendering.txt
rm -rf $output_figure_path
export DISPLAY=:99
Xvfb :99 -screen 0 640x480x24 &
python3 src/rendering_utils/img_renderer.py --input_dir $visual_obj_path --output_dir $output_figure_path > $log_path/rendering.txt
