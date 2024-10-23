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
        print(f"Loaded program from {pkl_file}")
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



    prompt = f"""You are a Python developer specializing in PyTorch API testing. Your task is to generate a Python test program for a given PyTorch API using fuzz testing techniques. The program will compare CPU and GPU responses to identify potential errors.



Please create a Python test program based on this API signature. Follow these guidelines:

1. Use fuzz testing to compare CPU and GPU responses.
2. Write the output to variables named "cpu_output" and "gpu_output".
3. Include the number of parameters in your code as "num_of_parameters".
4. Do not define or give values to the parameters. Instead, use "param1", "param2", etc., as placeholders.
5. Add comments to indicate what the parameters are, their expected types, and bounds at the end of the python code.
6. Provide only the Python code without any explanations.
7. Place the code in a markdown codeblock.

Before writing the code, analyze the API signature in <analysis> tags:

1. Identify the function name
2. For each parameter:
   - List its name
   - Note its type
   - Mention any default values or if it's optional
3. Identify the return type(s)
4. Consider the number of parameters and how this affects the test structure
5. Think about potential edge cases or specific requirements of the API
6. Plan the structure of the test program:
   a. Import necessary libraries (torch)
   b. Create input tensor(s) on CPU
   c. Move tensor(s) to GPU
   d. Perform the API operation on CPU
   e. Perform the same operation on GPU
   f. Include the number of parameters

Please generate the Python test program based on this analysis. Remember to follow the guidelines provided earlier.
Here is the PyTorch API signature you need to test:

<api_signature>
torch.matrix_exp(input)
</api_signature>
    """

    assistant_response = f"""<analysis>
1. Function name: torch.matrix_exp

2. Return type: Tensor

3. Potential edge cases:
   - Input tensor with very large or very small values
   - Input tensor with complex numbers
   - Square vs. non-square matrices

4. Test program structure:
   a. Import torch
   b. Create input tensor on CPU
   c. Move tensor to GPU
   d. Perform matrix_exp operation on CPU
   e. Perform matrix_exp operation on GPU
   f. Include number of parameters

5. Number of parameters: 5 (including dtype)

</analysis>

```python
import torch

x = torch.randn(param1, param2, dtype=param3)
y = x.cuda()

cpu_output = torch.matrix_exp(torch.log(x * param4 - param5)) # on CPU

gpu_output = torch.matrix_exp(torch.log(y * param4 - param5)) # on GPU

num_of_parameters=5

# Parameters:
#   - param1: an integer representing the first dimension of the input tensor
#   - param2: an integer representing the second dimension of the input tensor
#   - param3: dtype parameter, optional, specifies the data type of the input tensor
#   - param4: scalar value used in the calculation
#   - param5: scalar value used in the calculation
```"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates Python code using PyTorch."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_response},
        {"role": "user", "content": f"""Please generate the Python test program based on this analysis. Remember to follow the guidelines provided earlier.
Here is the PyTorch API signature you need to test:

<api_signature>
{api}
</api_signature>"""}
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
