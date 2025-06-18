import os
import requests
import base64
import json
import time
from mimetypes import guess_type
from tqdm import tqdm
# from parse_sequence import parse_sequence
# from parse_visual import run_parallel
# from parse_image import render_file
from call_openai import setup_client, call_openai
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image-folder-path', type=str, default='exp/figures/test', help='Path to the input folder')
parser.add_argument('--out-path', type=str, default='data/raw', help='Path to the output file')
args = parser.parse_args()
file_path = args.image_folder_path
out_path = args.out_path

client, deployment_name = setup_client()
call_client = call_openai

def local_image_to_data_url(image_path):
    # Encode a local image into data URL
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream' 
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

def call_model_1(prompt, image_path):
    message_text = [
        {"role":"system","content":"You are an AI assistant that helps people find information."},
        {"role":"user","content":[
            {
                "type": "text",
                "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {"url": local_image_to_data_url(image_path)}
            }
        ]}
    ]
    return call_client(client, deployment_name, message_text)

def call_model_2(prompt1, image_path, output1, prompt2):
    message_text = [
        {"role":"system","content":"You are an AI assistant that helps people find information."},
        {"role":"user","content":[
            {
                "type": "text",
                "text": prompt1
            },
            {
            "type": "image_url",
            "image_url": {"url": local_image_to_data_url(image_path)}
            }
        ]},
        {"role":"assistant","content":output1},
        {"role":"user","content":prompt2}
    ]
    return call_client(client, deployment_name, message_text)

files = [f for f in os.listdir(args.image_folder_path) if os.path.isfile(os.path.join(args.image_folder_path, f))]  
files.sort()  
results = []
for filename in tqdm(files):
    time.sleep(0.5)
    output1 = None
    output2 = None
    image_path = os.path.join(file_path, filename)  
    # Send request
    prompt1 = """Propose a series of questions about the 3D shape and give the answers. The first question should ask for a detailed description and others should focus on the specific geometric properties, number, size proportions and positional relationship, and other details."""
    prompt2 = """Based on the dialogue, please give a final description of the 3D shape. No more than 70 words."""
    while output1 is None or str(output1).startswith("I'm sorry"):
        try:
            output1 = call_model_1(prompt1, image_path)
        except requests.RequestException as e:  
            print(f"Request failed: {e}")
            time.sleep(1)  
            output1 = None  
    while output2 is None or str(output2).startswith("I'm sorry"):
        try:
            output2 = call_model_2(prompt1, image_path, output1, prompt2)
        except requests.RequestException as e:  
            print(f"Request failed: {e}")
            time.sleep(1)  
            output2 = None  

    result = {
        "pic_name":filename,
        "questions": output1,
        "description":output2
    }
    results.append(result)

with open(out_path, 'w+', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)