import re
import json
import pandas as pd

def extract_usage_info(file_content):
    # Regular expressions to extract the values
    completion_pattern = r'completion_tokens=(\d+)'
    prompt_pattern = r'prompt_tokens=(\d+)'
    total_pattern = r'total_tokens=(\d+)'
    cached_pattern = r'cached_tokens=(\d+)'
    
    # Find the values
    completion_tokens = int(re.search(completion_pattern, file_content).group(1))
    prompt_tokens = int(re.search(prompt_pattern, file_content).group(1))
    total_tokens = int(re.search(total_pattern, file_content).group(1))
    cached_tokens = int(re.search(cached_pattern, file_content).group(1))
    
    return {
        'completion_tokens': completion_tokens,
        'prompt_tokens': prompt_tokens,
        'total_tokens': total_tokens,
        'cached_tokens': cached_tokens
    }

def extract_program_info(json_content):
    data = json.loads(json_content)
    usage = data['usage']
    
    return {
        'completion_tokens': usage['completion_tokens'],
        'prompt_tokens': usage['prompt_tokens'],
        'total_tokens': usage['total_tokens']
    }

def process_all_files():
    # Lists to store results
    usage_results = []
    program_results = []
    
    # Process usage files
    print("Processing usage files...")
    for i in range(0, 99):
        filename = f'parameter_API_files/usage_{i}.txt'
        try:
            with open(filename, 'r') as file:
                content = file.read()
                usage_info = extract_usage_info(content)
                usage_info['file_number'] = i
                usage_results.append(usage_info)
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Process usage files
    print("Processing usage files...")
    for i in range(0, 99):
        filename = f'program_usage/usage_{i}.txt'
        try:
            with open(filename, 'r') as file:
                content = file.read()
                usage_info = extract_usage_info(content)
                usage_info['file_number'] = i
                program_results.append(usage_info)
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Convert results to DataFrames
    usage_df = pd.DataFrame(usage_results)
    program_df = pd.DataFrame(program_results)
    
    # Save to CSV
    usage_df.to_csv('usage_summary.csv', index=False)
    program_df.to_csv('program_summary.csv', index=False)
    
    # Print summary statistics
    print("\n=== Usage Files Summary Statistics ===")
    print(usage_df.describe())    
    print("\n=== Program Files Summary Statistics ===")
    print(program_df.describe())
    
    return usage_df, program_df

if __name__ == "__main__":
    try:
        usage_df, program_df = process_all_files()
        print("\nData successfully extracted and saved to 'usage_summary.csv' and 'program_summary.csv'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")