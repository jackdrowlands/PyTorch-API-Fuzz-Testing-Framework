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

# PyTorch API Fuzz Testing Framework
# Main runner script that generates test programs, executes them with various parameters,
# and records the results for analysis. Handles concurrent execution, caching, and 
# resource management for efficient testing.

def run_or_load_pytorch_code_with_params(code: str, params: List[str], id: int) -> List[dict]:
    """
    Run or load the results of running PyTorch code with parameters.
    
    Args:
        code: The PyTorch code to execute
        params: List of parameter sets for testing
        id: Unique identifier for this test run
        
    Returns:
        List of dictionaries containing test results
        
    Note:
        This function implements caching - if results for this id already exist,
        they are loaded from disk instead of re-running tests.
    """
    pkl_file = f'result_parts/result_{id}.pkl'

    # Check if results are already cached on disk
    if os.path.exists(pkl_file):
        print(f"Loaded results from {pkl_file}")
        with open(pkl_file, 'rb') as f:
            results = pickle.load(f)
        return results

    # Run the PyTorch code with the parameters
    results = run_pytorch_code_with_params(code, params, id)

    # Save the results to a pickle file for future runs
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f)

    return results


def run_pytorch_code_with_params(code: str, params_list: List[List[str]], id : int) -> List[dict]:
    """
    Run PyTorch code with a list of parameters and return the results.
    
    Args:
        code: The PyTorch code to execute
        params_list: List of parameter sets for testing
        id: Unique identifier for this test run
        
    Returns:
        List of dictionaries containing test results with keys:
        - params: Parameter set index
        - result: Whether CPU and GPU outputs match
        - output: Raw program output
        - error: Error message if execution failed
    """
    results = []
    i = 0
    for params in params_list:
        i += 1
        # Create a dictionary with the parameters
        # Convert empty strings to None values
        param_dict = {f'param{i+1}': param if param != '' else None for i, param in enumerate(params)}

        
        # Create a temporary Python file with the code and parameters
        with open(f"temp_code/temp_code_{id}.py", 'w') as f:
            # Set up imports and memory limit (4GB)
            f.write(f"""import torch
import json
import math
inf = math.inf
import resource
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, -1))
""")
            # Write parameter definitions
            for key, value in param_dict.items():
                f.write(f"{key} = {value}\n")
            # Write the test code
            f.write(code)
            # Add output conversion function
            f.write("""

def safe_convert_to_list(tensor):
    if isinstance(tensor, (bool, str, int, float, type(None))):
        return tensor
    return tensor.cpu().tolist()
print(json.dumps({'cpu_output': safe_convert_to_list(cpu_output), 'gpu_output': safe_convert_to_list(gpu_output.cpu())}))""")

        # Run the code in a separate process with resource limits (90 second timeout)
        try:
            result = subprocess.run([sys.executable, f"temp_code/temp_code_{id}.py"], 
                                    capture_output=True, text=True, timeout=90)
            
            

            # Process successful execution with output
            if result.stdout.strip():
                try:
                    # Parse JSON output and compare CPU vs GPU results
                    output = json.loads(result.stdout)
                    comparison = torch.allclose(torch.tensor(output['cpu_output']), 
                                        torch.tensor(output['gpu_output']), 
                                        rtol=1e-2, atol=1e-3, equal_nan=True)
                    
                    # Record the result
                    results.append({
                        'params': i,
                        'result': f"CPU and GPU outputs are {'equal' if comparison else 'not equal'}",
                        'output': result.stdout,
                        'error': result.stderr if result.returncode != 0 else None
                    })
                    # Save mismatched results for further analysis
                    if (not comparison):
                        print(f"CPU and GPU outputs are not equal for program {id} with parameters {i}")
                        print(f"Parameters: {i}")
                        # Save the code and parameters to a file for debugging
                        with open(f'mismatch_files/mismatch_{id}_{i}.py', 'w') as f:
                            f.write(code)
                            for key, value in param_dict.items():
                                f.write(f"{key} = {value}\n")
                except json.JSONDecodeError as e:
                    print(f"Error parsing output JSON: {str(e)}")
                    print(f"Raw output: {result.stdout}")
            else:
                # Handle case with no stdout (likely error)
                results.append({
                        'params': i,
                        'result': None,
                        'output': None,
                        'error': result.stderr
                    })

            
        except subprocess.TimeoutExpired:
            # Handle execution timeout (90 seconds)
            print("CPU time limit exceeded")
            results.append({
                'params': i,
                'result': None,
                'output': None,
                'error': "CPU time limit exceeded"
            })
        except Exception as e:
            # Handle other exceptions
            print(f"Error running PyTorch code: {str(e)}")
            results.append({
                'params': i,
                'result': None,
                'output': None,
                'error': str(e)
            })

    return results

def parameter_comparison_test(test_code: str, num_params: int, model: str, param_name: str, param_values: List[float]):
    """
    Run tests with different parameter values and return the results.
    
    Args:
        test_code: PyTorch code to test
        num_params: Number of parameters in the test code
        model: LLM model to use for parameter generation
        param_name: Name of the parameter to vary
        param_values: Different values to test for the specified parameter
        
    Returns:
        List of dictionaries with test results for each parameter value
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
    
    Args:
        results: List of dictionaries containing test results
        filename: Path to save the CSV file
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def process_single_api(i, api, num_apis):
    """
    Process a single API test case and save results to a separate file.
    
    This function is the core worker for parallel processing. It generates a test program
    for the given API, creates parameters, runs the test, and saves results.
    
    Args:
        i: Index of the API test case
        api: The PyTorch API to test
        num_apis: Total number of APIs being tested
        
    Returns:
        Index of the processed API
    """
    result_file = f'result_parts/result_{i}.pkl'
    
    # If result already exists, skip processing
    if os.path.exists(result_file):
        return i
    
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
            results = [{
                'program_id': i+1,
                # 'code': test_program["code"],
                'params': result['params'],
                'result': result['result'],
                'output': result['output'],
                'error': result['error']
            } for result in test_results]
        else:
            results = [{
                'program_id': i+1,
                # 'code': test_program["code"],
                'params': None,
                'result': None,
                'output': '',
                'error': 'Failed to generate program'
            }]
    except Exception as e:
        results = [{
            'program_id': i+1,
            # 'code': None,
            'params': None,
            'result': None,
            'output': '',
            'error': f'Exception: {str(e)}'
        }]

    # Save results to file
    os.makedirs('result_parts', exist_ok=True)
    with open(result_file, 'wb') as f:
        pickle.dump(results, f)
    
    return i

def main(num_programs: int = 1584, num_apis: int = 1, start_id: int = 0, max_workers: int = 12):
    """
    Generate, run, and record results for multiple test programs using multithreading.
    Results are saved in parts and combined at the end.
    
    This is the main entry point for the PyTorch API fuzzing framework. It handles:
    - Directory setup
    - API loading 
    - Parallel test execution with progress tracking
    - ETA calculation and display
    
    Args:
        num_programs: Number of test programs to run
        num_apis: Number of APIs to include in each test
        start_id: Starting index for test programs (for resuming interrupted runs)
        max_workers: Maximum number of parallel worker threads
    """
    import time
    from datetime import timedelta

    # Create results directory if it doesn't exist
    os.makedirs('result_parts', exist_ok=True)
    os.makedirs('temp_code', exist_ok=True)
    os.makedirs('mismatch_files', exist_ok=True)
    os.makedirs('parameter_files', exist_ok=True)
    os.makedirs('program_files', exist_ok=True)
    os.makedirs('parameter_API_files', exist_ok=True)
    os.makedirs('program_API_files', exist_ok=True)
    os.makedirs('program_usage', exist_ok=True)

    # Read in a list of PyTorch APIs
    with open('api_def_torch.txt', 'r') as f:
        apis = f.read().splitlines()

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
                completed_i = future.result()
                completed_tasks += 1
                thread_safe_print(f"Completed processing program {completed_i+1}")
                thread_safe_print(update_eta())
            except Exception as e:
                thread_safe_print(f"Error processing program {i+1}: {str(e)}")
                completed_tasks += 1
                thread_safe_print(update_eta())

    total_time = time.time() - start_time
    print(f"\nResults have been written to results.pkl")
    print(f"Total execution time: {format_time_remaining(total_time)}")

if __name__ == "__main__":
    main()