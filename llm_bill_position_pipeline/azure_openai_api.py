from openai import AzureOpenAI

def inference_azure(prompt, **kwargs):
    # gets the API Key from environment variable AZURE_OPENAI_API_KEY
    client = AzureOpenAI(
        api_key =kwargs['api_key'],
        api_version =kwargs['api_version'],
        azure_endpoint =kwargs['azure_endpoint'],
    )

    completion = client.chat.completions.create(
        model=kwargs['model'],
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return completion.choices[0].message.content.strip()
