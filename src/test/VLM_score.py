import os
import requests
import base64
import json
import time
import argparse
from mimetypes import guess_type
from tqdm import tqdm
import re

from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider

scope = "api://trapi/.default"
credential = get_bearer_token_provider(AzureCliCredential(),scope)

api_version = '2024-12-01-preview'
# deployment_name = 'gpt-4.1-mini_2025-04-14'
deployment_name = 'gpt-4o_2024-08-06'
instance = '<trapi/path>' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly)
endpoint = f'https://trapi.research.microsoft.com/{instance}'

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=credential,
    api_version=api_version,
)

def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

def ask_gpt(image_path, prompt):
    image_url = local_image_to_data_url(image_path)
    message_text = [
            {"role": "system", "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}
        ]

    completion = client.chat.completions.create(
            model=deployment_name,
            messages=message_text,)
    output = completion.choices[0].message.content
    return output

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-path', type=str, default='data/sl_data/test.jsonl', help='Path to the JSONL file containing test data')
    parser.add_argument('--name', type=str, default='original_seq', help='Run name of the testee')
    parser.add_argument('--figure-dir', type=str, default='exp/figures')
    parser.add_argument('--save-path', type=str, default='exp/evals', help='Target folder to save the results')
    parser.add_argument('--repetition', type=int, default=5, help='Number of repetitions for each image')
    args = parser.parse_args()

    results = []
    jsonl_path = args.test_path
    name = args.name
    figures_dir = f"{args.figure_dir}/{name}/"
    save_path = f"{args.save_path}/{name}.jsonl"

    with open(jsonl_path, 'r+') as file:
        test_data = json.load(file)
    repetition = args.repetition
    results = []
    for i in tqdm(range(len(test_data[:800]))):
        item = test_data[i]
        for j in range(repetition):
            img_num = i * repetition + j
            image_name = f"{img_num:06d}.png"
            image_path = os.path.join(figures_dir, image_name)
            if os.path.exists(image_path):
                description = item['description']
                try:
                    score = ask_gpt(image_path, f"The following is a text description of a 3D CAD figure and an image of a CAD instance. Measure if the figure corresponds to the given description, and give a score in the scale of 10. Only return the score. Do not comment on issues such as texture, smoothness and colors.\n description:{description}\n")

                    # "The following is an original image of a CAD instance, a text description on editing and an image of the edited result. Measure if the figure corresponds to the given description, and give a score in the scale of 10. Only return the score. Do not comment on issues such as texture, smoothness and colors.\n description:{description}\n"
                except Exception as e:
                    print(img_num)
                    print(e)
                    score = -1
                result = {
                    "index": img_num,
                    "gpt_score": score
                }
                results.append(result)
                with open(save_path, 'w+') as file:
                    json.dump(results, file, indent=4)