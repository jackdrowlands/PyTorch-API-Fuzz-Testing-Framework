import random
import requests
import os
import json
from typing import List, Union
import re

def generate_test_program(
    num_apis: int = 3,
    model: str = "nousresearch/hermes-3-llama-3.1-405b:free",
    max_tokens: int = 500
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

    # List of common PyTorch APIs
    pytorch_apis = [
        "torch.tensor", "torch.rand", "torch.zeros", "torch.ones",
        "torch.cat", "torch.stack", "torch.matmul", "torch.mm",
        "torch.sum", "torch.mean", "torch.max", "torch.min",
        "torch.relu", "torch.sigmoid", "torch.tanh", "torch.softmax",
        "torch.nn.Linear", "torch.nn.Conv2d", "torch.nn.RNN", "torch.nn.LSTM"
    ]

    prompt = f"""Generate a Python test program using {num_apis} random PyTorch APIs. 

    Use the following APIs (you can use each more than once if needed):
    {', '.join(random.sample(pytorch_apis, num_apis))}

    Provide only the Python code without any explanations in a markdown codeblock.

    The method of testing is fuzz testing using CPU compared to GPU responses to determine if there was an error.
    Please include both CPU and GPU sections, writing the output to "cpu_output" and "gpu_output".
    Additionally include the number of parameters in your code as "num_of_parameters".
    Please do not define the parameters, only include them as "param`n`".

    Here is an example program:
    ```python
    import torch

    x = torch.randn(param1, param2, dtype=param3)
    print("Intermediate: ", torch.log(x * 2 - 1)) # Intermediate
    cpu_output = torch.matrix_exp(torch.log(x * 2 - 1)) # on CPU
    x = x.cuda()
    gpu_output = torch.matrix_exp(torch.log(x * 2 - 1)) # on GPU

    num_of_parameters=3
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
        
        content = result['choices'][0]['message']['content']
        print(content)
        
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
        
        return {
            "code": code,
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

# Example usage
test_program = generate_test_program(num_apis=3)
print(test_program)