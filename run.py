import torch
import sys
from io import StringIO
from typing import List
import csv
import json
import os
import resource
import signal
import subprocess

from RestrictedPython import compile_restricted, safe_builtins
from parameters import create_fuzz_test_parameters, create_or_load_fuzz_test_parameters
from programs import generate_test_program, generate_or_load_test_program

def run_pytorch_code_with_params(code: str, params_list: List[List[str]]) -> List[dict]:
    results = []
    for params in params_list:
        param_dict = {f'param{i+1}': param for i, param in enumerate(params)}
        
        # Create a temporary Python file with the code and parameters
        with open('temp_code.py', 'w') as f:
            f.write(f"import torch\n")
            for key, value in param_dict.items():
                f.write(f"{key} = {value}\n")
            f.write(code)
            f.write("\nprint(json.dumps({'cpu_output': cpu_output.tolist(), 'gpu_output': gpu_output.cpu().tolist()}))")

        # Run the code in a separate process with resource limits
        try:
            result = subprocess.run([sys.executable, 'temp_code.py'], 
                                    capture_output=True, text=True, timeout=90)
            
            

            if result.stdout.strip():
                try:
                    output = json.loads(result.stdout)
                    comparison = torch.allclose(torch.tensor(output['cpu_output']), 
                                        torch.tensor(output['gpu_output']), 
                                        rtol=1e-5, atol=1e-8)
                    results.append({
                        'params': param_dict,
                        'result': f"CPU and GPU outputs are {'equal' if comparison else 'not equal'}",
                        'output': result.stdout,
                        'error': result.stderr if result.returncode != 0 else None
                    })
                except json.JSONDecodeError as e:
                    print(f"Error parsing output JSON: {str(e)}")
                    print(f"Raw output: {result.stdout}")
            else:
                print("Empty output from executed code")
                results.append({
                        'params': param_dict,
                        'result': None,
                        'output': None,
                        'error': result.stderr
                    })

            
        except subprocess.TimeoutExpired:
            results.append({
                'params': param_dict,
                'result': None,
                'output': None,
                'error': "CPU time limit exceeded"
            })
        except Exception as e:
            results.append({
                'params': param_dict,
                'result': None,
                'output': None,
                'error': str(e)
            })

    return results

def parameter_comparison_test(test_code: str, num_params: int, model: str, param_name: str, param_values: List[float]):
    """
    Run tests with different parameter values and return the results.
    """
    repetitions = 10
    results = []

    # Create a folder for parameter files if it doesn't exist
    params_folder = "parameter_files"
    os.makedirs(params_folder, exist_ok=True)

    for param_value in param_values:
        error_count = 0
        api_error_count = 0
        total_runs = 0

        # Use param_value as sets_per_repetition if that's the parameter being tested
        sets_per_repetition = param_value if param_name == 'sets_per_repetition' else 10

        for rep in range(repetitions):
            # Generate or load parameters for each parameter value and repetition
            params_file = os.path.join(params_folder, f"params_{param_name}_{param_value}_rep_{rep}.json")
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    content = f.read()
                    if content.startswith("API Error:"):
                        print(f"Skipping {params_file} due to previous API error")
                        api_error_count += sets_per_repetition
                        continue
                    params_list = json.loads(content)
            else:
                # max tokens is 400 or 20*sets_per_repetition 
                max_tokens = 20*sets_per_repetition
                params_list = create_fuzz_test_parameters(test_code, num_params=num_params, num_sets=sets_per_repetition, model=model, **{param_name: param_value}, max_tokens=max_tokens)
                if isinstance(params_list, list):
                    with open(params_file, 'w') as f:
                        json.dump(params_list, f)
                else:
                    error_message = f"API Error: {params_list}"
                    print(f"{error_message} at {param_name} {param_value}, repetition {rep}")
                    with open(params_file, 'w') as f:
                        f.write(error_message)
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

def main(num_programs: int = 10, num_apis: int = 3):
    """
    Generate, run, and record results for multiple test programs.
    """
    results = []

    for i in range(num_programs):
        print(f"Generating and running program {i+1}/{num_programs}")
        
        test_program = generate_or_load_test_program(num_apis=num_apis)
        if isinstance(test_program["code"], str) and not test_program["code"].startswith("Error"):
            params = create_or_load_fuzz_test_parameters(id=i, code_to_test=test_program["code"], num_params=test_program["num_of_parameters"])
            test_results = run_pytorch_code_with_params(test_program["code"], params)
    
            for result in test_results:
                results.append({
                    'program_id': i+1,
                    'code': test_program["code"],
                    'params': result['params'],
                    'result': result['result'],
                    'output': result['output'],
                    'error': result['error']
                })
        else:
            results.append({
                'program_id': i+1,
                'code': test_program["code"],
                'params': None,
                'result': None,
                'output': '',
                'error': 'Failed to generate program'
            })


    # Write results to CSV
    with open('test_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['program_id', 'code', 'params', 'result', 'output', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results have been written to test_results.csv")

if __name__ == "__main__":
    main()