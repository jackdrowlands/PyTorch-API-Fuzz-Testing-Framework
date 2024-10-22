import random
import requests
import os
import json
from typing import List, Union
import re
import csv
import pickle

def generate_or_load_test_program(id : int, api: str, num_apis: int = 3, model: str = "openai/gpt-4o-mini", max_tokens: int = 1000) -> Union[str, List[str]]:
    pkl_file = f'program_files/program_{id}.pkl'
    
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            program = pickle.load(f)
        return program
            
    
    # If no matching program found, generate a new one
    program = generate_test_program(id, api, num_apis, model, max_tokens)
    
    # Save the new program to pkl file
    with open(pkl_file, 'wb') as f:
        pickle.dump(program, f)
    
    return program


def generate_test_program(
    id: int,
    api: str,
    num_apis: int = 1,
    model: str = "google/gemini-pro-1.5-exp",
    max_tokens: int = 1000
) -> Union[str, List[str]]:
    """
    Generate a test program using random PyTorch APIs using an LLM.

    Args:
        num_apis (int): Number of PyTorch APIs to include in the test program.
        model (str): The LLM model to use.
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        Union[str, List[str]]: The generated test program as a string, or an error message.
    """



    prompt = f"""Generate a Python test program to test this PyTorch API. Here is the API signature:
    {api}


    Provide only the Python code without any explanations in a markdown codeblock.

    The method of testing is fuzz testing using CPU compared to GPU responses to determine if there was an error.
    Please include both CPU and GPU sections, writing the output to "cpu_output" and "gpu_output".
    Additionally include the number of parameters in your code as "num_of_parameters".
    Please do not define or give values to the parameters, only include them as "param`n`".

    Here is an example program:
    ```python
    import torch

    x = torch.randn(param1, param2, dtype=param3)
    y = x.cuda()

    cpu_output = torch.matrix_exp(torch.log(x * param4 - param5)) # on CPU

    gpu_output = torch.matrix_exp(torch.log(y * param4 - param5)) # on GPU

    num_of_parameters=5
    ```
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates Python code using PyTorch."},
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "provider": {
            "require_parameters": True
        }
    }


    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()
        result = response.json()
        print(result)

        # Save full response to json file
        with open(f'program_API_files/program_{id}.json', 'w') as f:
            json.dump(result, f)
        
        content = result['choices'][0]['message']['content']
        
        if content is None:
            return {
                "code": f"An error occurred: {content}",
                "num_of_parameters": None
            }
        
        # Extract code from markdown codeblock
        if "```python" in content and "```" in content:
            code = content.split("```python")[1].split("```")[0].strip()
        else:
            # if it isn't in a codeblock, use all of the code.
            code = content.strip()
        
        # Extract num_of_parameters
        num_of_parameters = None
        match = re.search(r'num_of_parameters\s*=\s*(\d+)', code)
        if match:
            num_of_parameters = int(match.group(1))
        
        # delete all definitions of the parameters
        cleaned_code = re.sub(r"\bparam\w*\s*=\s*.*?(?=\n|\Z)","",code)
        
        return {
            "code": cleaned_code,
            "num_of_parameters": num_of_parameters
        }

    except requests.exceptions.RequestException as e:
        return {
            "code": f"Error: {str(e)}",
            "num_of_parameters": None
        }
    except json.JSONDecodeError:
        return {
            "code": "Error: Invalid JSON response",
            "num_of_parameters": None
            }
    except KeyError as e:
        return {
            "code": f"Error: Missing key in response: {str(e)}",
            "num_of_parameters": None
            }
    except Exception as e:
        return {
            "code": f"Unexpected error: {str(e)}",
            "num_of_parameters": None
            }
