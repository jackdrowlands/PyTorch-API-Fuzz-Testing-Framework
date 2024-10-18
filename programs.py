import random
import requests
import os
import json
from typing import List, Union

def generate_test_program(
    num_apis: int = 3,
    model: str = "meta-llama/llama-3.1-8b-instruct",
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

    Provide only the Python code without any explanations.
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
        
        if content is None:
            return f"An error occurred: {content}"
        
        code_lines = content.split('\n')
        code = '\n'.join(line for line in code_lines if not line.strip().startswith('#'))
        
        return code.strip()

    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON response"
    except KeyError as e:
        return f"Error: Missing key in response: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Example usage
test_program = generate_test_program(num_apis=3)
print(test_program)