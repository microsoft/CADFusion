gdown --id 1so_CCGLIhqGEDQxMoiR--A4CQk4MjuOp
unzip cad_data.zip

# convert data into sequence and save in json
mkdir data
mkdir data/raw
python3 src/data_preprocessing/convert.py --in-path cad_data/train_deduplicate_s.pkl --out-path data/raw/train.json
python3 src/data_preprocessing/convert.py --in-path cad_data/val.pkl --out-path data/raw/val.json
python3 src/data_preprocessing/convert.py --in-path cad_data/test.pkl --out-path data/raw/test.json

# render the image for each entry in order to retrieve textual information by captioning:
mkdir exp
mkdir exp/visual_objects
mkdir exp/figures
for file in test val train; do
    python3 src/rendering_utils/parser.py --in-path data/raw/$file.json --out-path exp/visual_objects/$file
    timeout 180 python3 src/rendering_utils/parser_visual.py --data_folder exp/visual_objects/$file

    export DISPLAY=:99
    Xvfb :99 -screen 0 640x480x24 &
    python3 src/rendering_utils/img_renderer.py --input_dir exp/visual_objects/$file --output_dir exp/figures/$file
done

# caption the images to generate descriptions
mkdir data/sl_data
python3 src/data_preprocessing/captioning.py --image-folder-path exp/figures/train --out-path data/sl_data/train.json
python3 src/data_preprocessing/captioning.py --image-folder-path exp/figures/val --out-path data/sl_data/val.json
python3 src/data_preprocessing/captioning.py --image-folder-path exp/figures/test --out-path data/sl_data/test.json