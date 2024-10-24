import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def load_results(file_path='results.pkl'):
    print(f"Loading results from {file_path}...")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def analyze_results(results):
    # Convert results to a DataFrame
    print("Converting results to DataFrame...")
    df = pd.DataFrame(results)
    
    # Basic statistics
    print("Basic statistics:")
    total_programs = df['program_id'].nunique()
    total_tests = len(df)
    failed_tests = df['error'].notnull().sum()
    cpu_gpu_mismatch = df['result'].str.contains('not equal', na=False).sum()
    
    print(f"Total programs: {total_programs}")
    print(f"Total tests: {total_tests}")
    print(f"Failed tests: {failed_tests}")
    print(f"Failure rate: {failed_tests/total_tests:.2%}")
    print(f"CPU-GPU mismatches: {cpu_gpu_mismatch}")
    print(f"CPU-GPU mismatch rate: {cpu_gpu_mismatch/total_tests:.2%}")
    
     # Improved error analysis
    def extract_error_message(error):
        if not isinstance(error, str):
            return str(error)
        
        # Split the error message into lines
        lines = error.split('\n')
        
        # Look for the actual error message (usually the last line or the line starting with specific error types)
        for line in reversed(lines):
            if any(err in line for err in ['RuntimeError:', 'ValueError:', 'TypeError:', 'IndexError:', 'Exception:']):
                return line.strip()
        
        # If no specific error found, return the last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        return error

    # Create error DataFrame with specific error messages
    error_df = df[df['error'].notnull()].copy()
    error_df['error_type'] = error_df['error'].apply(extract_error_message)
    error_counts = error_df['error_type'].value_counts()
    
    print("\nTop error types with counts:")
    for error_type, count in error_counts.head(20).items():
        print(f"\n{count:4d} occurrences: {error_type}")
    
    # Plot error distribution
    plt.figure(figsize=(15, 8))
    error_counts.head(10).plot(kind='bar')
    plt.title('Distribution of Top 10 Error Types')
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.close()

    # Print every id with the error: "Failed to generate program"
    print("\nProgram IDs with the error: 'Failed to generate program'")
    for idx, row in error_df[error_df['error_type'] == 'Failed to generate program'].iterrows():
        print(row['program_id'])
    
    # Analyze CPU-GPU mismatches
    mismatch_df = df[df['result'].str.contains('not equal', na=False)].copy()
    
    print(f"\nTotal CPU-GPU mismatches: {len(mismatch_df)}")
    
    # Analyze APIs involved in mismatches
    mismatch_api_counts = Counter()
    for code in mismatch_df['code'].dropna():
        apis = [line.split('=')[0].strip() + ' = ' + line.split('(')[0].strip() for line in code.split('\n') if '=' in line and 'torch.' in line]
        mismatch_api_counts.update(apis)
    
    print("\nTop 10 APIs involved in CPU-GPU mismatches:")
    for api, count in mismatch_api_counts.most_common(10):
        print(f"{api}: {count}")
    
    # Plot API involvement in mismatches
    plt.figure(figsize=(12, 6))
    pd.Series(mismatch_api_counts).nlargest(10).plot(kind='bar')
    plt.title('Top 10 APIs Involved in CPU-GPU Mismatches')
    plt.xlabel('API')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('mismatch_api_involvement.png')
    plt.close()
    
    # Analyze parameter distributions for mismatches
    mismatch_param_dfs = []
    for params in mismatch_df['params'].dropna():
        mismatch_param_dfs.append(pd.DataFrame(params, index=[0]))
    mismatch_param_df = pd.concat(mismatch_param_dfs, ignore_index=True)
    
    print("\nParameter statistics for CPU-GPU mismatches:")
    print(mismatch_param_df.describe())
    
    # Analyze parameter value types
    param_types = mismatch_param_df.applymap(lambda x: str(type(x).__name__))
    type_counts = param_types.apply(pd.Series.value_counts).fillna(0)
    print("\nParameter value types in mismatches:")
    print(type_counts)
    
    # Plot parameter value types
    type_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Parameter Value Types in CPU-GPU Mismatches')
    plt.xlabel('Parameter')
    plt.ylabel('Count')
    plt.legend(title='Value Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('mismatch_parameter_types.png')
    plt.close()
    
    # # Print mismatched programs and parameters
    # print("\nMismatched programs and parameters (sorted by difference):")
    # for idx, row in mismatch_df.iterrows():
    #     print(f"\nProgram ID: {row['program_id']}")
    #     print("Code:")
    #     print(row['code'])
    #     print("Parameters:")
    #     for key, value in row['params'].items():
    #         print(f"{key}: {value}")
    #     print("-" * 50)
    
    return df

if __name__ == "__main__":
    results = load_results()
    df = analyze_results(results)
    print("\nAnalysis complete. Check the generated PNG files for visualizations.")