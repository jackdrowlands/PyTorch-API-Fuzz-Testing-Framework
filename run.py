import concurrent
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
import math
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
import threading

from RestrictedPython import compile_restricted, safe_builtins
from parameters import create_fuzz_test_parameters, create_or_load_fuzz_test_parameters
from programs import generate_test_program, generate_or_load_test_program

def run_or_load_pytorch_code_with_params(code: str, params: List[str], id: int) -> List[dict]:
    """
    Run or load the results of running PyTorch code with parameters.
    """
    pkl_file = f'result_parts/results_{id}.pkl'

    if os.path.exists(pkl_file):
        print(f"Loaded results from {pkl_file}")
        with open(pkl_file, 'rb') as f:
            results = pickle.load(f)
        return results

    # Run the PyTorch code with the parameters
    results = run_pytorch_code_with_params(code, params, id)

    # Save the results to a pickle file
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f)

    return results


def run_pytorch_code_with_params(code: str, params_list: List[List[str]], id : int) -> List[dict]:
    results = []
    for params in params_list:
        # Create a dictionary with the parameters
        # if the parameter is em
        param_dict = {f'param{i+1}': param if param != '' else None for i, param in enumerate(params)}

        
        # Create a temporary Python file with the code and parameters
        with open(f"temp_code/temp_code_{id}.py", 'w') as f:
            f.write(f"import torch\nimport json\nimport math\ninf = math.inf\nimport resource\nresource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, -1))\n")
            for key, value in param_dict.items():
                f.write(f"{key} = {value}\n")
            f.write(code)
            f.write("\nprint(json.dumps({'cpu_output': cpu_output.tolist(), 'gpu_output': gpu_output.cpu().tolist()}))")

        # Run the code in a separate process with resource limits
        try:
            result = subprocess.run([sys.executable, f"temp_code/temp_code_{id}.py"], 
                                    capture_output=True, text=True, timeout=90)
            
            

            if result.stdout.strip():
                try:
                    output = json.loads(result.stdout)
                    comparison = torch.allclose(torch.tensor(output['cpu_output']), 
                                        torch.tensor(output['gpu_output']), 
                                        rtol=1e-3, atol=1e-6)
                    # print(f"CPU and GPU outputs are {'equal' if comparison else 'not equal'}")
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
                # Print the last 100 characters of the error message
                # print(f"Error running PyTorch code: {result.stderr[-100:]}")
                results.append({
                        'params': param_dict,
                        'result': None,
                        'output': None,
                        'error': result.stderr
                    })

            
        except subprocess.TimeoutExpired:
            print("CPU time limit exceeded")
            results.append({
                'params': param_dict,
                'result': None,
                'output': None,
                'error': "CPU time limit exceeded"
            })
        except Exception as e:
            print(f"Error running PyTorch code: {str(e)}")
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


def process_single_api(i, api, num_apis):
    """Process a single API test case"""
    # print(f"Generating and running program {i+1} with the following API: {api}")
    
    try:
        test_program = generate_or_load_test_program(id=i, api=api, num_apis=num_apis)
        if isinstance(test_program["code"], str) and not test_program["code"].startswith("Error"):
            num_param_sets = math.floor(90 / (test_program["num_of_parameters"] + 1))
            # print(f"Running program with {num_param_sets} parameter sets")
            params = create_or_load_fuzz_test_parameters(
                id=i, 
                code_to_test=test_program["code"], 
                num_params=test_program["num_of_parameters"], 
                num_sets=num_param_sets
            )
            test_results = run_or_load_pytorch_code_with_params(test_program["code"], params, id=i)
            return [{
                'program_id': i+1,
                'code': test_program["code"],
                'params': result['params'],
                'result': result['result'],
                'output': result['output'],
                'error': result['error']
            } for result in test_results]
        else:
            return [{
                'program_id': i+1,
                'code': test_program["code"],
                'params': None,
                'result': None,
                'output': '',
                'error': 'Failed to generate program'
            }]
    except Exception as e:
        print(f"Error processing API {api} (id={i}): {str(e)}")
        return [{
            'program_id': i+1,
            'code': None,
            'params': None,
            'result': None,
            'output': '',
            'error': f'Exception: {str(e)}'
        }]

def main(num_programs: int = 1584, num_apis: int = 1, start_id: int = 0, max_workers: int = 12):
    """
    Generate, run, and record results for multiple test programs using multithreading.
    """
    import time
    from datetime import timedelta

    # Read in a list of PyTorch APIs
    with open('api_def_torch.txt', 'r') as f:
        apis = f.read().splitlines()

    all_results = []
    
    # Create a lock for thread-safe printing
    print_lock = threading.Lock()
    
    # Track timing information
    start_time = time.time()
    completed_tasks = 0
    total_tasks = num_programs - start_id
    
    def thread_safe_print(*args, **kwargs):
        with print_lock:
            print(*args, **kwargs)

    def format_time_remaining(seconds):
        """Format remaining time in a human-readable countdown format"""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            if minutes == 0:
                return f"{hours} hour{'s' if hours != 1 else ''}"
            return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"

    def update_eta():
        current_time = time.time()
        elapsed_time = current_time - start_time
        if completed_tasks == 0:
            return "Calculating..."
        
        avg_time_per_task = elapsed_time / completed_tasks
        remaining_tasks = total_tasks - completed_tasks
        remaining_time = avg_time_per_task * remaining_tasks
        
        return (f"Progress: {completed_tasks}/{total_tasks} ({(completed_tasks/total_tasks)*100:.1f}%) "
                f"| Remaining time: {format_time_remaining(remaining_time)} "
                f"| Elapsed: {format_time_remaining(elapsed_time)}")

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_api = {
            executor.submit(process_single_api, i, apis[i % len(apis)], num_apis): i 
            for i in range(start_id, num_programs)
        }
        
        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_api):
            i = future_to_api[future]
            try:
                results = future.result()
                all_results.extend(results)
                completed_tasks += 1
                thread_safe_print(f"Completed processing program {i+1}")
                thread_safe_print(update_eta())
            except Exception as e:
                thread_safe_print(f"Error processing program {i+1}: {str(e)}")
                completed_tasks += 1
                thread_safe_print(update_eta())

    # Write results to pickle
    with open('results.pkl', 'wb') as pickle_file:
        pickle.dump(all_results, pickle_file)

    total_time = time.time() - start_time
    print(f"\nResults have been written to results.pkl")
    print(f"Total execution time: {format_time_remaining(total_time)}")

if __name__ == "__main__":
    main()