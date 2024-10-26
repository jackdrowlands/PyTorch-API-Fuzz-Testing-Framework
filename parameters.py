from openai import OpenAI
import os
from typing import Optional, Dict, Any, List, Union
import requests
import json
from pydantic import BaseModel, Field, conlist
import csv
import re
import pickle

def create_json_schema(num_parameter_sets, num_params_per_set):
    properties = {}
    
    for i in range(1, num_parameter_sets + 1):
        parameter_set_key = f"parameter_set_{i}"
        parameter_set_properties = {}
        
        for j in range(1, num_params_per_set + 1):
            param_key = f"param{j}"
            parameter_set_properties[param_key] = {
                "type": "string",
                "properties": {},
                "additionalProperties": False,
                "description": f"Parameter {j} for test case {i}."
            }
        
        properties[parameter_set_key] = {
            "type": "object",
            "properties": parameter_set_properties,
            "required": list(parameter_set_properties.keys()),
            "additionalProperties": False,
            "description": f"Set of parameters for test case {i}"
        }
    
    json_schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
        "description": "Collection of parameter sets for fuzz testing"
    }
    
    return {
        "name": "fuzz_test_parameters",
        "schema": json_schema,
        "strict": True
    }

def extract_responses(completion_response, id):
    # Extract the content from the response
    content = completion_response.choices[0].message.content
    
    with open(f'parameter_API_files/completion_{id}.txt', 'w') as f:
        f.write(content)
    
    with open(f'parameter_API_files/usage_{id}.txt', 'w') as f:
        f.write(str(completion_response.usage))

    # Initialize the list to hold all parameter sets
    all_responses = []
    
    try:
        # Parse the entire JSON object
        data = json.loads(content)
        
        # Iterate through each parameter set
        for i in range(1, len(data) + 1):
            param_set_key = f"parameter_set_{i}"
            if param_set_key in data:
                param_set = data[param_set_key]
                
                # Initialize a list for this parameter set
                param_set_responses = []
                
                # Iterate through each parameter in the set
                for j in range(1, len(param_set) + 1):
                    param_key = f"param{j}"
                    if param_key in param_set:
                        # Convert the parameter value to a string and add it to the list
                        param_set_responses.append(str(param_set[param_key]))
                
                # Add this parameter set's responses to the main list
                all_responses.append(param_set_responses)
    
    except json.JSONDecodeError:
        print("Error: Failed to parse the JSON data.")
        return []
    
    return all_responses

def create_or_load_fuzz_test_parameters(
    id : int,
    code_to_test: str,
    num_params: int,
    num_sets: int = 10,
    model: str = "gpt-4o-mini",
    max_tokens: int = 16384,
    **kwargs: Any
) -> Union[List[List[str]], str]:
    csv_file = f'parameter_files/fuzz_test_parameters_{id}_{num_params}_{num_sets}.csv'
    
    if os.path.exists(csv_file):
        print(f"Loaded existing parameters from {csv_file}")
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            return list(reader)
    
    # If no matching parameters found, generate new ones
    parameters = create_fuzz_test_parameters(code_to_test, num_params, num_sets, model, max_tokens, id, **kwargs)
    
    # Save the new parameters to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(parameters)
    
    return parameters


def create_fuzz_test_parameters(
    code_to_test: str,
    num_params: int, # Number of parameters to generate
    num_sets: int = 10, # Number of sets of parameters to generate
    model: str = "gpt-4o-mini", # Model to use
    max_tokens: int = 16384, # Maximum number of tokens to generate
    id: int = 0, # ID for the generated parameters
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
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    # Prepare the messages
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": f"""You are an AI assistant tasked with generating parameter sets for fuzz testing a PyTorch API call. Your goal is to create unique and novel sets of parameters that will test the robustness and error handling of the API.

Instructions:
1. Generate unique and novel parameter sets that will test various edge cases and potential error conditions.
2. Ensure that each parameter is of the correct data type for the API call.
3. Explicity define the data type for each parameter. For example: "param1": "str('fro')", "param2": "int(42)", "param3": "float(3.14)", "param4": "bool(True)", "param5": "torch.float32".
4. Do not use single quotes around numeric, type, or boolean parameters. For example: "param1": "torch.float32".
5. Ensure that the program will not exceed 4GB of memory usage when using these parameters.
6. Keep tensor dimensions under 1000.
7. Make sure all values are within the allowed range for each parameter.
8. Create exactly the number of parameter sets specified, ensuring each set is unique.
9. Remember that these parameters will be injected into the code in the order they are generated.
10. Do not reference any later parameters in the code. For example, do not use "param2" in the definition of "param1".

When generating parameters for torch.randn() or random tensor operations:
        ONLY use these datatypes:
        - torch.float32 (default)
        - torch.float64
        
        DO NOT use these datatypes with random operations:
        - torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64
        - torch.bool, torch.char, torch.byte, torch.short, torch.long

Review the API code that needs to be tested:

<api_code>
{code_to_test}
</api_code>

You need to generate <num_params>{num_params}</num_params> parameters for each set, and create a total of <num_sets>{num_sets}</num_sets> unique sets.
"""}
    ]
    json_format = create_json_schema(num_parameter_sets=num_sets, num_params_per_set=num_params)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": json_format
        },
        max_tokens=max_tokens,
        **kwargs
    )

    # print(f"Cached tokens for parameters: {completion.usage.prompt_tokens_details.cached_tokens}")

    return extract_responses(completion, id)
    # except Exception as e:
    #     print(str(e))
    #     return f"Error: {str(e)}"