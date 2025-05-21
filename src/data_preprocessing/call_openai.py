from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider

import time

def setup_client():
    scope = "api://trapi/.default"
    credential = get_bearer_token_provider(AzureCliCredential(), scope)

    api_version = '2024-12-01-preview'
    deployment_name = 'gpt-4o_2024-08-06'
    instance = 'gcr/shared/' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly)
    endpoint = f'https://trapi.research.microsoft.com/{instance}'

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=api_version,
    )
    return client, deployment_name


def call_openai(client, deployment, prompt):
    output = None
    while output is None:
        try:
            time.sleep(0.5)
            completion = client.chat.completions.create(
                model = deployment,
                messages = prompt,
            )
            output = completion.choices[0].message.content
        except Exception as e:  
            print("API error:", e)  
            time.sleep(1)
            output = None
    return output