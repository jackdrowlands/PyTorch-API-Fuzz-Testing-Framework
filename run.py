import torch
import sys
from io import StringIO
from typing import List, Any
import csv
from parameters import create_fuzz_test_parameters

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
        namespace.update(param_dict)

        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Execute the code with the current parameters
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

def temperature_comparison_test(test_code: str, num_params: int, model: str):
    temperatures = [round(t * 0.1, 1) for t in range(21)]  # 0.0 to 2.0 in 0.1 increments
    repetitions = 10
    results = []

    for temp in temperatures:
        error_count = 0
        for _ in range(repetitions):
            params_list = create_fuzz_test_parameters(test_code, num_params=num_params, num_sets=1, model=model, temperature=temp)
            if isinstance(params_list, str):
                error_count += 1
                continue
            
            test_results = run_pytorch_code_with_params(test_code, params_list)
            if test_results[0]['error'] is not None:
                error_count += 1

        results.append({
            'temperature': temp,
            'error_count': error_count,
            'success_rate': (repetitions - error_count) / repetitions
        })

    return results

def save_results_to_csv(results: List[dict], filename: str):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['temperature', 'error_count', 'success_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    test_code = """
import torch
x = torch.tensor([param1, param2, param3])
cpu_output = 1 / torch.clamp(x, min=param4, max=param5)
x = x.cuda()
gpu_output = 1 / torch.clamp(x, min=param4, max=param5)
"""

    model = "nousresearch/hermes-3-llama-3.1-405b:extended"
    results = temperature_comparison_test(test_code, num_params=5, model=model)
    
    save_results_to_csv(results, 'temperature_comparison_results.csv')

    print("Results saved to temperature_comparison_results.csv")
    
    # Print a summary of the results
    print("\nTemperature Comparison Summary:")
    for result in results:
        print(f"Temperature: {result['temperature']:.1f}, Error Count: {result['error_count']}, Success Rate: {result['success_rate']:.2f}")
