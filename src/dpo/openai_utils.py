import os
import base64
import time
import json

from mimetypes import guess_type
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

END_POINT = '<endpoint>'
MODEL_NAME = 'gpt-4o_2024-08-06'
API_VER = '2024-02-01'

def local_image_to_data_url(image_path):
    # Encode a local image into data URL
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream' 
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

def ask_gpt_on_figure(data, _, __):
    endpoint = END_POINT
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )
    deployment_name = MODEL_NAME

    client = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=endpoint,
        api_version=API_VER
    )
    description = data['description']
    data_scale = 10
    measurement = 'if the figure corresponds to the given description'

    prompt = 'The following is a text description of a 3D CAD figure and an image of a CAD instance. ' +\
        f'Measure {measurement}, and give a score in the scale of {data_scale}. Do not comment on issues such as texture, smoothness and colors' +\
        f'\n description: {description}\n'
    image_path = data['figure_path']
    response = client.chat.completions.create(
        model=deployment_name, 
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': local_image_to_data_url(image_path)}},  
            ]}
        ]
    )
    time.sleep(3)
    return(response.choices[0].message.content)


def ask_gpt(data, _, __):
    endpoint = END_POINT
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )
    deployment_name = MODEL_NAME

    client = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=endpoint,
        api_version=API_VER
    )
    description = data['gpt_label']

    prompt = 'The following is an evaluation of an CAD object.' +\
        f'\n evaluation: {description}\n' +\
        'Extract the integer score of the evaluation. The score is between 0 to 10. Return the number only.'
    
    response = client.chat.completions.create(
        model=deployment_name, 
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': [            
                {'type': 'text', 'text': prompt},
            ]}
        ]
    )
    # print(response.choices[0].message.content)
    time.sleep(3)
    return(response.choices[0].message.content)