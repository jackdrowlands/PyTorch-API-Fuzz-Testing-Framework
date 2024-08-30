import torch
import sys
from io import StringIO
from typing import List, Any

def run_pytorch_code_with_params(code: str, params_list: List[List[str]]) -> List[dict]:
    results = []

    # Create a namespace for executing the code
    namespace = {'torch': torch}

    # Compile the code
    try:
        compiled_code = compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return [{'params': {}, 'result': None, 'output': None, 'error': f'Syntax error in code: {str(e)}'}]

    for params in params_list:
        # Prepare the parameter dictionary
        param_dict = {f'param{i+1}': eval(param, namespace) for i, param in enumerate(params)}
        
        # Add param_dict to the namespace
        namespace.update(param_dict)  # Add this line

        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Execute the code with the current parameters
            exec(compiled_code, namespace)
            
            # Call the last defined function with the parameters
            last_function = list(namespace.values())[-1]
            if callable(last_function):
                result = last_function(**param_dict)
            else:
                result = "No callable function found in the code"

            # Capture any printed output
            printed_output = sys.stdout.getvalue()

            results.append({
                'params': param_dict,
                'result': result,
                'output': printed_output.strip(),
                'error': None
            })
        except Exception as e:
            results.append({
                'params': param_dict,
                'result': None,
                'output': None,
                'error': str(e)
            })
        finally:
            # Restore stdout
            sys.stdout = old_stdout

    return results

# Example usage
if __name__ == "__main__":
    from parameters import create_fuzz_test_parameters

    test_code = """
import torch

x = torch.randn(param1, param2, dtype=param3)
print("Intermediate: ", torch.log(x * 2 - 1)) # Intermediate
output = torch.matrix_exp(torch.log(x * 2 - 1)) # on CPU
print("Output CPU: ", output)
x = x.cuda()
output = torch.matrix_exp(torch.log(x * 2 - 1)) # on GPU
print("Output GPU: ", output)
    """

    # Generate parameters using the create_fuzz_test_parameters function
    params_list = create_fuzz_test_parameters(test_code, num_params=3, model="nousresearch/hermes-3-llama-3.1-405b:extended", temperature=0)

    # Run the code with the generated parameters
    results = run_pytorch_code_with_params(test_code, params_list)

    # Print the results
    for i, result in enumerate(results):
        print(f"\nTest {i + 1}:")
        print(f"Parameters: {result.get('params', 'No parameters found')}")
        print(f"Result: {result.get('result', 'No result')}")
        print(f"Output: {result.get('output', 'No output')}")
        print(f"Error: {result.get('error', 'No error')}")
