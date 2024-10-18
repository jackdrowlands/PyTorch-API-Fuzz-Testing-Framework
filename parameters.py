from openai import OpenAI
import os
from typing import Optional, Dict, Any, List, Union
import requests
import json

def create_fuzz_test_parameters(
    code_to_test: str,
    num_params: int, # Number of parameters to generate
    num_sets: int = 10, # Number of sets of parameters to generate
    model: str = "nousresearch/hermes-3-llama-3.1-405b:free", # Model to use
    max_tokens: int = 500, # Maximum number of tokens to generate
    **kwargs: Any # Additional parameters to pass to the model
) -> Union[List[List[str]], str]:
    """
    Generate fuzz test parameters using an AI model.
    
    This function sends a request to an AI model to generate sets of parameters
    for fuzz testing a given PyTorch API call.

    Returns:
        List[List[str]]: A list of lists of parameters.
        str: An error message if the API call fails.
    """

    # Prepare the messages
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": f"""You are tasked with generating 10 unique and novel sets of 3 parameters for a PyTorch API call. The purpose is to fuzz test the API call.

Your task is to create parameters that will test the robustness and error handling of this API call. Please consider the requirements of the API call and the allowed range of values for each parameter.
Your output should be structured in CSV format, containing only the parameters. Do not include explanations or justifications in the output.

Now, based on the API code provided and the guidelines above, please generate 10 unique and novel sets of 3 parameters for fuzz testing. Output your response in CSV format inside <parameters> tags.
<api_code>
import torch

x = torch.randn(param1, param2, dtype=param3)
print("Intermediate: ", torch.log(x * 2 - 1)) # Intermediate
cpu_output = torch.matrix_exp(torch.log(x * 2 - 1)) # on CPU
x = x.cuda()
gpu_output = torch.matrix_exp(torch.log(x * 2 - 1)) # on GPU
</api_code>
num_params = 3
num_sets = 10
"""},
{"role": "assistant", "content": "<parameters>\n5,5,torch.float32\n1000,1000,torch.float64\n0,0,torch.int32\n-1,-1,torch.float16\n1000000,1,torch.bfloat16\n10,10,torch.complex64\n2,3,torch.complex128\n100,100,torch.int8 \n10000,10000,torch.int16\n1,1000000,torch.int64\n</parameters>"},
{"role": "user", "content": f"""Now, based on the API code provided and the guidelines above, please generate {num_sets} unique and novel sets of {num_params} parameters for fuzz testing. Output your response in CSV format inside <parameters> tags.
<api_code>
{code_to_test}
</api_code>
num_params = {num_params}
num_sets = {num_sets}
"""}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "provider": {
            "require_parameters": True
        },
        **kwargs
    }
    # Add max_tokens if provided
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    # Set up the headers
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",  # Use environment variable
        "Content-Type": "application/json"
    }

    # print("API Request Details:")
    # print(f"URL: https://openrouter.ai/api/v1/chat/completions")
    # print(f"Headers: {json.dumps(headers, indent=2)}")
    # print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        # Set up the openrouter client
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()
        # Parse the JSON response
        result = response.json()
        
        # Extract the content from the response
        content = result['choices'][0]['message']['content']
        
        # Parse the JSON content
        if content is None:
            return f"An error occurred: {content}"
        # Check if there is a <parameters> tag in the response
        if '<parameters>' in content:
            # Extract the CSV content from the response
            csv_content = content.split('<parameters>')[1].split('</parameters>')[0].strip()
            # Parse the CSV content
            parameters = [row.split(',') for row in csv_content.split('\n') if row.strip()]
        else:
            return f"Error: No <parameters> tag found in the response:\n{content}"

        # Check for the correct number of parameters
        if len(parameters) != num_sets:
            return f"Error: Incorrect number of parameter sets. Expected {num_sets}, got {len(parameters)}\n{content}"
        else:
            print("Parameters: ", parameters)
            return parameters

    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON response"
    except KeyError as e:
        return f"Error: Missing key in response: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"