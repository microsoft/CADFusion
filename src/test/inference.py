import argparse
import json
import random

from huggingface_hub import login
from tqdm import tqdm
from utils import MAX_LENGTH, prepare_model_and_tokenizer

login()

random.seed(0)

def conditional_sample(args):
    model, tokenizer = prepare_model_and_tokenizer(args)
    
    model.eval()
    with open(args.in_path, 'r', encoding='utf-8') as file:  
        data = json.load(file)
        
    print(data[0])
    data = [item for item in data if item['description'] != 'null']

    global_count=0
    responses = []  
    if args.full:
        data=data
    else:
        random.shuffle(data)
        data = data[:args.sample_len]
    
    for item in tqdm(data):
        prompts = []
        for _ in range(args.num_samples):
            prompt = 'Below is a description of a 3D shape:\n'
            prompt += item['description']  
            prompt += '\nGenerate a Computer-Aided Design(CAD) command sequence of the 3D shape:\n'     

            prompts.append(prompt)

        outputs = []

        while len(outputs) < args.num_samples:
            batch_prompts = prompts[len(outputs) : len(outputs) + args.batch_size]

            batch = tokenizer(
                list(batch_prompts),
                return_tensors="pt",
            )
            batch = {k: v.cuda() for k, v in batch.items()}

            generate_ids = model.generate(
                **batch,
                do_sample=True,
                max_new_tokens=MAX_LENGTH,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.3,
            )

            gen_strs = tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            outputs.extend(gen_strs)
            print(f"Generated {len(outputs)}/{args.num_samples}samples.")

            for prompt, output in zip(prompts, outputs):
                result = {
                    'index': global_count,
                    # 'pic_name': item['pic_name'],
                    'ground_truth': item['command_sequence'],
                    'description': item['description'],
                    'prompt': prompt,
                    'output': output[len(prompt):]
                }
                if 'original_seq' in item.keys():
                    result['original_seq'] = item['original_seq']
                responses.append(result)
                global_count += 1

        with open(args.out_path, "w+") as f:
            json.dump(responses, f, indent=4)

                
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="llama3")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--sample-len", type=int, default=100)
    parser.add_argument("--pretrained-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--in-path", type=str, default="test_description.json")
    parser.add_argument("--out-path", type=str, default="cad_samples.jsonl")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--device-map", type=str, default='auto')
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--full", action="store_true", default=False)
    args = parser.parse_args()

    conditional_sample(args)
