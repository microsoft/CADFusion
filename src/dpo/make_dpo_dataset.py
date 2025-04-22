import json
import os
import time
import argparse

from openai_utils import ask_gpt_on_figure, ask_gpt
from llava_utils import ask_llm, ask_llm_on_figure, restart_model
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-path", type=str, required=True)
    parser.add_argument("--figure-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--score-only", action="store_true", default=False)
    parser.add_argument("--gpt", action="store_true", default=False)
    args = parser.parse_args()
    
    source_path = args.source_data_path
    folder_path = args.figure_path
    save_path = args.save_path
    num_samples = args.num_samples
    device=f'cuda:{args.gpu}'
    if args.gpt:
        func1, func2 = ask_gpt_on_figure, ask_gpt
        model = None
        processor = None
    else:
        func1, func2 = ask_llm_on_figure, ask_llm
        model, processor = restart_model(device)
    
    with open(source_path, 'r') as f:
        test_data = json.load(f)
    
    ####### Stage 1 #######
    # for model generations that are able to render pictures,
    # ask gpt to rate the generation quality.
    for data in tqdm(test_data):
        file_id = str(data['index']).zfill(6)
        file = None
        for f in os.listdir(folder_path):
            if f.startswith(file_id):
                file = folder_path + f
                data['figure_path'] = file
                error_cnt = 0
                while 1:
                    try:
                        data['gpt_label'] = func1(data, model, processor)
                        break
                    except Exception as e:
                        print(e)
                        if args.gpt:
                            time.sleep(3)
                        else:
                            if error_cnt == 5:
                                exit()
                            model, processor = restart_model(device)
                    error_cnt += 1
    with open(save_path, 'w+') as f:
        json.dump(test_data, f, indent=4)   
    
    with open(save_path, 'r') as f:
        test_data = json.load(f)
    ####### Stage 2 #######
    # clean up the dataset to summarize the generation quality estimation to a numerical score, and
    # remove the failed ones, i.e. the generations that cannot render
    for data in tqdm(test_data):
        if "gpt_label" in data.keys():
            error_cnt = 0
            while 1:
                try:
                    score = func2(data, model, processor)
                    print(score)
                    break
                except Exception as e:
                    print(e)
                    if args.gpt:
                        time.sleep(3)
                    else:
                        if error_cnt == 5:
                            exit()
                        model, processor = restart_model(device)
                error_cnt += 1
            try:
                data['gpt_score'] = int(score)
            except:
                print(f'ERROR: {score}')
                pass    
                
    saved_data = [data for data in test_data if 'gpt_score' in data.keys()]     
    with open(save_path, 'w+') as f:
        json.dump(saved_data, f, indent=4)   
    
    if args.score_only:
        exit()
    
    ####### Stage 3 #######
    # 1. group up the scored generations by their description: we do not compare 
    #    generation results that come from different origin prompts
    temp_data = []
    max_idx = test_data[-1]['index']
    sample_size = max_idx // num_samples + 1
    # a. select if any above 6

    # for i in range(sample_size):
    #     next_sample = test_data[i*num_samples:(i+1)*num_samples]
    #     next_sample = [item for item in next_sample if 'gpt_score' in item.keys()]
    #     above_score = [item['gpt_score'] >= 6 for item in next_sample]
    #     if any(above_score):
    #         temp_data.extend(next_sample)
    # temp_data = [data for data in temp_data if 'gpt_score' in data.keys()] 
    
    # b. select if avg above 6

    # for i in range(sample_size):
    #     next_sample = test_data[i*num_samples:(i+1)*num_samples]
    #     next_sample = [item for item in next_sample if 'gpt_score' in item.keys()]
    #     if len(next_sample) == 0:
    #         continue
    #     scores = sum(item['gpt_score'] for item in next_sample) / len(next_sample)
    #     if scores >= 6:
    #         temp_data.extend(next_sample)
    # temp_data = [data for data in temp_data if 'gpt_score' in data.keys()]

    # c. select if individual above 6
    test_data = saved_data
    for item in test_data:
        if 'gpt_score' not in item.keys():
            continue
        if item['gpt_score'] >= 6:
            temp_data.append(item)
    print(test_data[-1]['index'], max_idx)
    
    grouped = [[] for _ in range(max_idx)]
    for item in temp_data:
        idx = item['index']
        grouped[idx // num_samples].append(item)
    grouped = [item for item in grouped if len(item) > 0]
    
    # 2. within each group, make pairs where the chosens have higher score than the rejected ones.
    # TODO: find a way to balance the data generated from each group
    final_data = []
    for group in grouped:
        for item1 in group:
            for item2 in group:
                if item2['gpt_score'] > item1['gpt_score']:
                    info_dict = {
                        "description": item1['description'],
                        "prompt": item1['prompt'],
                        "chosen": item2['output'],
                        "rejected": item1['output']
                        }
                    final_data.append(info_dict)
                    # uncomment this break if you do not want too many data.
                    # break
            

    with open(save_path, 'w+') as f:
        json.dump(final_data, f, indent=4)   