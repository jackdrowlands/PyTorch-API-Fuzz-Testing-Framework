import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import glob

def load_and_analyze_results(files_pattern='result_parts/result_*.pkl'):
    total_programs = 1583
    total_tests = 0
    failed_tests = 0
    cpu_gpu_mismatch = 0
    error_counts = Counter()
    mismatch_api_counts = Counter()
    mismatch_param_dfs = []
    
    # Iterate over each file
    for file_path in glob.glob(files_pattern):
        print(f"Loading results from {file_path}...")
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
        except pickle.UnpicklingError:
            print(f"Warning: {file_path} is corrupted or incomplete and will be skipped.")
            continue
        
        # Convert results to a DataFrame for analysis
        df = pd.DataFrame(results)

        if 'error' not in df.columns:
            df['error'] = None
        if 'result' not in df.columns:
            df['result'] = ""
        if 'output' not in df.columns:
            df['output'] = ""
        if 'params' not in df.columns:
            df['params'] = [{}]
        
        # Update basic statistics
        total_tests += len(df)
        failed_tests += df['error'].notnull().sum()
        cpu_gpu_mismatch += df['result'].str.contains('not equal', na=False).sum()
        
        # Analyze error types
        error_df = df[df['error'].notnull()].copy()
        error_df['error_type'] = error_df['error'].apply(extract_error_message)
        error_counts.update(error_df['error_type'].value_counts())
        
        # Analyze CPU-GPU mismatches
        mismatch_df = df[df['result'].str.contains('not equal', na=False)].copy()
        mismatch_df = mismatch_df[~mismatch_df['output'].str.contains('nan', na=False)]
        
        for code in mismatch_df['output'].dropna():
            apis = [line.split('=')[0].strip() + ' = ' + line.split('(')[0].strip() for line in code.split('\n') if '=' in line and 'torch.' in line]
            mismatch_api_counts.update(apis)
        
        # Collect parameters from mismatches
        for params in mismatch_df['params'].dropna():
            mismatch_param_dfs.append(pd.DataFrame([params]))
    
    # Print summary statistics
    print("\nTotal programs:", total_programs)
    print("Total tests:", total_tests)
    print("Failed tests:", failed_tests)
    print("Failure rate:", f"{failed_tests/total_tests:.2%}")
    print("CPU-GPU mismatches:", cpu_gpu_mismatch)
    print("CPU-GPU mismatch rate:", f"{cpu_gpu_mismatch/total_tests:.2%}")

    # Display top error types
    print("\nTop error types with counts:")
    for error_type, count in error_counts.most_common(20):
        print(f"{count:4d} occurrences: {error_type}")
    
    # Plot error distribution
    plt.figure(figsize=(15, 8))
    pd.Series(error_counts).head(10).plot(kind='bar')
    plt.title('Distribution of Top 10 Error Types')
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.close()
    
    # Convert mismatch_api_counts to a numeric Series
    mismatch_api_series = pd.Series(mismatch_api_counts).astype(int)

    # Check if there is data to plot
    if not mismatch_api_series.empty:
        # Plot API involvement in mismatches if data exists
        plt.figure(figsize=(12, 6))
        mismatch_api_series.nlargest(10).plot(kind='bar')
        plt.title('Top 10 APIs Involved in CPU-GPU Mismatches')
        plt.xlabel('API')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('mismatch_api_involvement.png')
        plt.close()
        print("Mismatch API involvement plot saved as 'mismatch_api_involvement.png'")
    else:
        print("No data available for mismatch API involvement plot.")


def extract_error_message(error):
    if not isinstance(error, str):
        return str(error)
    
    lines = error.split('\n')
    for line in reversed(lines):
        if any(err in line for err in ['RuntimeError:', 'ValueError:', 'TypeError:', 'IndexError:', 'Exception:']):
            return line.strip()
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return error

if __name__ == "__main__":
    load_and_analyze_results()
    print("\nAnalysis complete. Check the generated PNG files for visualizations.")
