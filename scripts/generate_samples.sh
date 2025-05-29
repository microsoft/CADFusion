train_data_path=data/sl_data/train.json
test_data_path=data/sl_data/test.json
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

model_path=exp/model_ckpt/$1
inference_path=exp/model_generation/$run_name.jsonl
visual_obj_path=exp/visual_objects/$run_name
output_figure_path=exp/figures/$run_name
log_path=exp/logs/$run_name

mkdir $log_path

echo "--------------------Inferencing--------------------" > $log_path/inference.txt
rm $inference_path
python3 src/test/inference.py --pretrained-path $model_path --in-path $data_path --out-path $inference_path --num-samples 5 --temperature $temperature --model-name llama3 > $log_path/inference.txt $3

echo "--------------------Parsing CAD objects--------------------" > $log_path/parsing_cad.txt
rm -rf $visual_obj_path
python3 src/rendering_utils/parser.py --in-path $inference_path --out-path $visual_obj_path > $log_path/parsing_cad.txt

echo "--------------------Parsing visual objects--------------------" > $log_path/parsing_visual.txt
python3 src/rendering_utils/parser_visual.py --data_folder $visual_obj_path > $log_path/parsing_visual.txt
python3 src/rendering_utils/ptl_sampler.py --in_dir $visual_obj_path --out_dir ptl > $log_path/sampling_ptl.out

echo "--------------------Rendering--------------------" > $log_path/rendering.txt
rm -rf $output_figure_path
export DISPLAY=:99
Xvfb :99 -screen 0 640x480x24 &
python3 src/rendering_utils/img_renderer.py --input_dir $visual_obj_path --output_dir $output_figure_path > $log_path/rendering.txt
