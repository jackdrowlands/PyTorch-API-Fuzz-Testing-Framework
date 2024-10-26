import random
import requests
import os
import json
from typing import List, Union
import re
import csv
import pickle
from openai import OpenAI

def generate_or_load_test_program(id : int, api: str, num_apis: int = 3, model: str = "gpt-4o-mini", max_tokens: int = 1000) -> Union[str, List[str]]:
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
    model: str = "gpt-4o-mini",
    max_tokens: int = 16384
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
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    prompt = f"""You are a Python developer specializing in PyTorch API testing. Your task is to generate a Python test program for a given PyTorch API using fuzz testing techniques. The program will compare CPU and GPU responses to identify potential errors.

Please create a Python test program based on this API signature. Follow these guidelines:

1. Use fuzz testing to compare CPU and GPU responses.
2. Write the output to variables named "cpu_output" and "gpu_output".
3. Include the number of parameters in your code as "num_of_parameters".
4. Do not define or give values to the parameters. Instead, use "param1", "param2", etc., as placeholders.
5. Add comments to indicate what the parameters are, their expected types, and bounds at the end of the python code. If there are few values, explicitly write out all the possible values. For dtypes, specify the possible data types.
6. Provide only the Python code without any explanations.
7. Place the code in a markdown codeblock.

There must be at least the following variables in your code:
- cpu_output
- gpu_output
- num_of_parameters

Before writing the code, analyze the API signature in <analysis> tags:

1. Identify the function name
2. Identify the return type(s)
3. Think about potential edge cases or specific requirements of the API
4. Plan the structure of the test program:
   a. Import necessary libraries (torch)
   b. Create input tensor(s) on CPU
   c. Move tensor(s) to GPU
   d. Perform the API operation on CPU
   e. Perform the same operation on GPU
   f. Include the number of parameters
5. Write the Python code based on the analysis in the code block.
    a. Use the provided API signature and placeholder parameters (param1, param2, etc.)
    b. Write the number of parameters at the end of the code.
    c. Add comments to explain the purpose of each parameter
    d. Include the expected type and range of each parameter


Please generate the Python test program based on this analysis. Remember to follow the guidelines provided earlier.
Here is the PyTorch API signature you need to test:

<api_signature>
torch.matrix_exp(input)
</api_signature>"""

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

</analysis>

```python
import torch

x = torch.randn(param1, param2, dtype=param3)
y = x.cuda()

cpu_output = torch.matrix_exp(torch.log(x * param4 - param5)) # on CPU

gpu_output = torch.matrix_exp(torch.log(y * param4 - param5)) # on GPU

num_of_parameters=5

# Parameters:
#   - param1: an integer representing the first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: an integer representing the second dimension of the input tensor (Must be a square tensor) : Range = [param1, param1] : Type = int
#   - param3: dtype parameter, optional, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16', 'torch.int8', 'torch.uint8', 'torch.int16', 'torch.int32', 'torch.int64', 'torch.bool', 'torch.complex64', 'torch.complex128'] : Type = torch.dtype
#   - param4: scalar value used in the calculation : Range = (-math.inf, math.inf) : Type = float
#   - param5: scalar value used in the calculation : Range = (-math.inf, math.inf)  : Type = float
```
"""
    prompt2=""""Please generate the Python test program based on this analysis. Remember to follow the guidelines provided earlier.
Here is the PyTorch API signature you need to test:

<api_signature>
torch.nn.functional.relu(input, inplace=False)
</api_signature>"""
    
    assistant_response2 = f"""<analysis>
1. Function name: torch.nn.functional.relu

2. Return type: Tensor

3. Potential edge cases:
   - Input tensor with negative values
   - Input tensor with zero values
   - Different input shapes and dimensions
   - Different data types
   - Inplace operation vs non-inplace

4. Test program structure:
   a. Import torch and torch.nn.functional
   b. Create input tensor on CPU
   c. Move tensor to GPU
   d. Perform relu operation on CPU
   e. Perform relu operation on GPU
   f. Include number of parameters
</analysis>

```python
import torch
import torch.nn.functional as F

x = torch.randn(param1, param2, param3, dtype=param4)
y = x.cuda()

cpu_output = F.relu(x, inplace=param5)  # on CPU
gpu_output = F.relu(y, inplace=param5)  # on GPU

num_of_parameters = 5

# Parameters:
#   - param1: first dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param2: second dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param3: third dimension of the input tensor : Range = [1, 10000] : Type = int
#   - param4: dtype parameter, specifies the data type of the input tensor : Range = ['torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16'] : Type = torch.dtype
#   - param5: boolean flag indicating whether to perform the operation in-place : Range = [True, False] : Type = bool
```"""
    
    final_prompt = f"""Please generate the Python test program based on this analysis. Remember to follow the guidelines provided earlier.
Here is the PyTorch API signature you need to test:

<api_signature>
{api}
</api_signature>"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates Python code using PyTorch."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_response},
        {"role": "user", "content": prompt2},
        {"role": "assistant", "content": assistant_response2},
        {"role": "user", "content": final_prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        # print(f"Cached tokens for program: {completion.usage.prompt_tokens_details.cached_tokens}")
        
        content = completion.choices[0].message.content

        # # Save full response to json file
        # with open(f'program_API_files/program_{id}.json', 'w') as f:
        #     json.dump(result, f)
        with open(f'program_usage/usage_{id}.txt', 'w') as f:
            f.write(str(completion.usage))
        
        # content = result['choices'][0]['message']['content']
        
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
        print(f"Error: {str(e)}")
        return {
            "code": f"Error: {str(e)}",
            "num_of_parameters": None
        }
    except json.JSONDecodeError:
        print("Error: Invalid JSON response")
        return {
            "code": "Error: Invalid JSON response",
            "num_of_parameters": None
            }
    except KeyError as e:
        print(f"Error: Missing key in response: {str(e)}")
        return {
            "code": f"Error: Missing key in response: {str(e)}",
            "num_of_parameters": None
            }
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {
            "code": f"Unexpected error: {str(e)}",
            "num_of_parameters": None
            }
