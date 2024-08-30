from openai import OpenAI
import os
from typing import Optional, Dict, Any, List

def create_fuzz_test_parameters(
    code_to_test: str,
    num_params: int, # Number of parameters to generate
    num_sets: int = 10, # Number of sets of parameters to generate
    model: str = "openai/gpt-4o", # Model to use
    temperature: float = 0.7, # Temperature of the model - how creative the model is
    max_tokens: Optional[int] = None, # Maximum number of tokens to generate
    top_p: float = 1.0, # Top P of the model - controls the randomness of the model
    frequency_penalty: float = 0.0, # Frequency penalty of the model - controls the frequency of the model
    presence_penalty: float = 0.0, # Presence penalty of the model - controls the presence of the model
    **kwargs: Any # Additional parameters to pass to the model
) -> List[List[str]]:
    # Prepare the messages
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": f"""You are tasked with generating {num_sets} unique and novel sets of {num_params} parameters for a PyTorch API call. The purpose is to fuzz test the API call. Here is the API code you should reference:
<api_code>
{code_to_test}
</api_code>

Your task is to create parameters that will test the robustness and error handling of this API call. Please consider the requirements of the API call and the expected range of values for each parameter.
Your output should be structured in CSV format, containing only the parameters. Do not include explanations or justifications in the output.

Here is an example of how your output might look:

Example 1:
num_params = 10
num_sets = 2
<parameters>
param11,param12,param13,param14,param15,param16,param17,param18,param19,param110
param21,param22,param23,param24,param25,param26,param27,param28,param29,param210
</parameters>
...

Now, based on the API code provided and the guidelines above, please generate {num_sets} unique and novel sets of {num_params} parameters for fuzz testing. Output your response in CSV format inside <parameters> tags."""}
    ]

    # Prepare the API call parameters
    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        **kwargs
    }
    # Add max_tokens if provided
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    try:
        # Set up the openrouter client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1"
        )
        # Make the API call
        response = client.chat.completions.create(**params)
        print("Response: ", response)
        # Check if there is a <parameters> tag in the response
        if '<parameters>' in response.choices[0].message.content:
            # Extract the CSV content from the response
            csv_content = response.choices[0].message.content.split('<parameters>')[1].split('</parameters>')[0].strip()
            # Parse the CSV content
            parameters = [row.split(',') for row in csv_content.split('\n') if row.strip()]
        else:
            return f"An error occurred: {response.choices[0].message.content}"

        # Check for the correct number of parameters
        if len(parameters) != num_sets:
            return f"An error occurred: {response.choices[0].message.content}"
        else:
            print("Parameters: ", parameters)
            return parameters

    except Exception as e:
        print("Error: ", e)
        return f"An error occurred: {str(e)}"
