import os
import base64
import time
import json
import requests
from mimetypes import guess_type
from transformers import pipeline
from transformers import LlavaNextProcessor
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import torch
from PIL import Image
dev='cuda:0'

# processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
# model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
# model.to(device)

def restart_model(device):  
    global dev
    dev = device
    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to(device)
    return model, processor

def ask_llm_on_figure(data, model, processor):
    """
    The layout of a typical data item
    {
        "index": 1,
        "pic_name": "000000_001_final.png",
        "ground_truth": "line,9,9 <curve_end> line,9,53 <curve_end> line,53,53 <curve_end> line,53,9 <curve_end> <loop_end> circle,31,29,31,20,35,25,27,25 <curve_end> <loop_end> circle,31,41,31,32,35,37,27,37 <curve_end> <loop_end> <face_end> <sketch_end> add,31,32,31,31,31,0,1,0,0,0,1,1,0,0,62,31,31 <extrude_end>",
        "description": "Create a rectangular panel with two circular through-holes centrally aligned on the vertical axis.",
        "prompt": "Below is a description of a 3D shape:\nCreate a rectangular panel with two circular through-holes centrally aligned on the vertical axis.\nGenerate a Computer-Aided Design(CAD) command sequence of the 3D shape:\n",
        "output": "line,se,9 <curve_end> line,ne,9 <curve_end> line,ne,53 <curve_end> line,se,53 <curve_end> <loop_end> circle,22,41,22, Twenty1 ,31,30,12,30 <curve_end> <loop_end> circle,40,21,40, Ten2 ,50,32,29,32 <curve_end> <loop_end> <face_end> <sketch_end> add,31,33,31,31,31,1,0,0,0,0,1,0,-1,0,62,31,31 <extrude_end>"
    },
    """
    url = data['figure_path']
    image = Image.open(url)
    description = data['description']
    # data_scale = 10
    # measurement = 'the degree of correspondence between them'

    prompt = 'You are a harsh grader for new CAD designers\' works. The following is a text description of a CAD figure that they designed and an image of a CAD instance.' +\
    f'\nDescription: {description}\n ' + \
    f'Comment on this work for \n '+\
    '1. If the overall shape remains correct; \n '+\
    '2. If the number of components are correct, especially the circular holes; \n '+\
    '3. If the distribution of the components are natural, i.e. they are not clustered together or collide with each other.\n'+\
    'After that, give a score out of 10. Do not comment on issues such as texture, smoothness and colors'

    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt",).to(dev, torch.float16)

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=256, pad_token_id=processor.tokenizer.eos_token_id)
    output = processor.decode(output[0], skip_special_tokens=True)
    idx = output.index('assistant\n')
    response = output[idx+10:]
    return(response)


def ask_llm(data, model, processor):
    description = data['gpt_label']

    prompt = 'The following is an evaluation of an CAD object.' +\
        f'\n evaluation: {description}\n' +\
        'Extract the integer score of the evaluation. The score is between 0 to 10. Return the number only.'
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="pt",).to(dev, torch.float16)

    output = model.generate(**inputs, max_new_tokens=16, pad_token_id=processor.tokenizer.eos_token_id)
    output = processor.decode(output[0], skip_special_tokens=True)
    idx = output.index('assistant\n')
    response = output[idx+10:]
    return(response)