import torch
import sys
from io import StringIO
from typing import List
import csv
import json
import os
from parameters import create_fuzz_test_parameters

def run_pytorch_code_with_params(code: str, params_list: List[List[str]]) -> List[dict]:
    """
    Execute PyTorch code with different sets of parameters and return the results.
    """
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
        namespace.update(param_dict)

        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Execute the code and compare CPU and GPU outputs
            exec(compiled_code, namespace)
            
            # Compare cpu_output and gpu_output
            cpu_output = namespace.get('cpu_output')
            gpu_output = namespace.get('gpu_output')
            
            if cpu_output is not None and gpu_output is not None:
                comparison = torch.allclose(cpu_output, gpu_output.cpu(), rtol=1e-5, atol=1e-8)
                result = f"CPU and GPU outputs are {'equal' if comparison else 'not equal'}"
            else:
                result = "CPU or GPU output not found in the code"

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

def parameter_comparison_test(test_code: str, num_params: int, model: str, param_name: str, param_values: List[float]):
    """
    Run tests with different parameter values and return the results.
    """
    repetitions = 10
    sets_per_repetition = 10
    results = []

    for param_value in param_values:
        error_count = 0
        api_error_count = 0
        total_runs = 0

        for rep in range(repetitions):
            # Generate or load parameters for each parameter value and repetition
            params_file = f"params_{param_name}_{param_value}_rep_{rep}.json"
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    params_list = json.load(f)
            else:
                params_list = create_fuzz_test_parameters(test_code, num_params=num_params, num_sets=sets_per_repetition, model=model, **{param_name: param_value}, max_tokens=400)
                if isinstance(params_list, list):
                    with open(params_file, 'w') as f:
                        json.dump(params_list, f)
                else:
                    print(f"API Error at {param_name} {param_value}, repetition {rep}: {params_list}")
                    api_error_count += sets_per_repetition
                    continue
            
            try:
                # Run the PyTorch code with the generated parameters
                test_results = run_pytorch_code_with_params("""
import torch
x = torch.tensor([param1, param2, param3])
cpu_output = 1 / torch.clamp(x, min=param4, max=param5)
gpu_output = 1 / torch.clamp(x, min=param4, max=param5)
""", params_list)
                error_count += sum(1 for result in test_results if result['error'] is not None)
                total_runs += len(test_results)
            except Exception as e:
                print(f"Error running PyTorch code at {param_name} {param_value}, repetition {rep}: {str(e)}")
                error_count += sets_per_repetition
                total_runs += sets_per_repetition

        # Calculate and store the success rate for each parameter value
        total_expected_runs = repetitions * sets_per_repetition
        success_rate = (total_expected_runs - error_count - api_error_count) / total_expected_runs

        results.append({
            param_name: param_value,
            'error_count': error_count,
            'api_error_count': api_error_count,
            'success_rate': success_rate
        })

    return results

def save_results_to_csv(results: List[dict], filename: str):
    """
    Save the test results to a CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    # Define the PyTorch code to be tested
    test_code = """
import torch
x = torch.tensor([param1, param2, param3])
cpu_output = 1 / torch.clamp(x, min=param4, max=param5)
x = x.cuda()
gpu_output = 1 / torch.clamp(x, min=param4, max=param5)
"""

    # Set the model
    model = "nousresearch/hermes-3-llama-3.1-405b"

    # Define parameters to test and their ranges
    params_to_test = {
        'temperature': [round(t * 0.1, 1) for t in range(16)],  # 0.0 to 1.5 in 0.1 increments
        'top_p': [round(p * 0.1, 1) for p in range(11)],  # 0.0 to 1.0 in 0.1 increments
        'top_k': list(range(0, 101, 10)),  # 0 to 100 in steps of 10
        'frequency_penalty': [round(f * 0.2 - 2, 1) for f in range(21)],  # -2.0 to 2.0 in 0.2 increments
        'presence_penalty': [round(p * 0.2 - 2, 1) for p in range(21)],  # -2.0 to 2.0 in 0.2 increments
        'repetition_penalty': [round(r * 0.1, 1) for r in range(10, 21)],  # 1.0 to 2.0 in 0.1 increments
        'min_p': [round(m * 0.1, 1) for m in range(11)],  # 0.0 to 1.0 in 0.1 increments
        'top_a': [round(a * 0.1, 1) for a in range(11)]  # 0.0 to 1.0 in 0.1 increments
    }

    # Run tests for each parameter
    for param_name, param_values in params_to_test.items():
        print(f"\nTesting {param_name}...")
        results = parameter_comparison_test(test_code, num_params=5, model=model, param_name=param_name, param_values=param_values)
        
        # Save the results to a CSV file
        csv_filename = f'{param_name}_comparison_results.csv'
        save_results_to_csv(results, csv_filename)
        print(f"Results saved to {csv_filename}")
        
        # Print a summary of the results
        print(f"\n{param_name.capitalize()} Comparison Summary:")
        for result in results:
            print(f"{param_name}: {result[param_name]:.2f}, Error Count: {result['error_count']}, API Error Count: {result['api_error_count']}, Success Rate: {result['success_rate']:.2f}")
