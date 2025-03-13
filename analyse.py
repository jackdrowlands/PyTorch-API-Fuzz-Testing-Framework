import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Any, Generator
from tqdm import tqdm
import csv
import json
import numpy as np

# PyTorch API Fuzz Testing Framework - Analysis Module
# This script processes test results to analyze performance, success rates,
# error patterns, and correlations between different test attributes.
# It generates visualizations and statistics to help understand test outcomes
# and identify patterns in API behavior.

# Directory and file path configurations
RESULTS_DIR = 'result_parts'        # Directory containing test result pickle files
PROGRAMS_DIR = 'program_files'      # Directory containing generated program pickle files
PARAMETERS_DIR = 'parameter_files'  # Directory containing parameter set CSV files
API_DEFINITION_FILE = 'api_def_torch.txt'  # File containing PyTorch API definitions
OUTPUT_CSV = 'analysis_report.csv'  # Output file for analysis results
OUTPUT_DIR = 'analysis_outputs'     # Directory for saving visualizations
nHead = 20                          # Number of top items to show in rankings

def calculate_success_by_parameter_count(program_data: Dict[int, Dict[str, Any]], results_stream: Generator[Dict[str, Any], None, None]) -> pd.DataFrame:
    """
    Calculate success rates grouped by number of parameters in each test.
    
    This function analyzes how test success rates correlate with parameter count,
    helping identify if APIs with more parameters have different success patterns.
    
    Args:
        program_data: Dictionary mapping program IDs to their metadata (including parameter count)
        results_stream: Generator yielding test result dictionaries
        
    Returns:
        DataFrame containing success rates for each parameter count
    """
    # Initialize counters for each parameter count
    param_success = defaultdict(lambda: {'Success': 0, 'Total': 0})
    
    # Process each test result
    for result in tqdm(results_stream, desc="Processing for parameter count analysis"):
        program_id = result.get('program_id')
        if program_id is None:
            continue
        
        # Get parameter count for this program
        num_params = program_data.get(program_id, {}).get('num_of_parameters')
        if num_params is None:
            continue
            
        # Update counters
        param_success[num_params]['Total'] += 1
        if result.get('result') == 'CPU and GPU outputs are equal':
            param_success[num_params]['Success'] += 1

    # Calculate success rates and prepare data for DataFrame
    data = []
    for num, counts in param_success.items():
        success_rate = (counts['Success'] / counts['Total']) * 100 if counts['Total'] > 0 else 0
        data.append({
            'Number_of_Parameters': num,
            'Success_Rate (%)': success_rate,
            'Total_Runs': counts['Total']
        })

    # Handle empty data case
    if not data:
        print("No data: parameter count analysis")
        return pd.DataFrame(columns=['Number_of_Parameters', 'Success_Rate (%)', 'Total_Runs'])

    # Convert to DataFrame and sort by parameter count
    df = pd.DataFrame(data).sort_values(by='Number_of_Parameters')
    return df


def visualize_success_by_parameter_count(df_success_param: pd.DataFrame, output_dir: str):
    """
    Create visualizations showing relationship between parameter count and success rate.
    
    Generates two plots:
    1. Line plot showing success rate for each parameter count
    2. Regression plot showing the trend line with statistical fit
    
    Args:
        df_success_param: DataFrame containing parameter counts and success rates
        output_dir: Directory to save generated visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate line plot of success rate vs parameter count
    plt.figure(figsize=(12,6))
    sns.lineplot(x='Number_of_Parameters', y='Success_Rate (%)', data=df_success_param, marker='o')
    plt.title('Success Rate vs. Number of Parameters')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Success Rate (%)')
    plt.xticks(df_success_param['Number_of_Parameters'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_vs_num_parameters.png'))
    plt.close()
    
    # Generate regression plot with trend line
    plt.figure(figsize=(12,6))
    sns.regplot(x='Number_of_Parameters', y='Success_Rate (%)', data=df_success_param, 
                scatter_kws={'s':100}, line_kws={'color':'red'})
    plt.title('Regression: Success Rate vs. Number of Parameters')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Success Rate (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_success_rate_vs_num_parameters.png'))
    plt.close()
    
    print("Generated success rate vs. number of parameters visualization.")

from scipy.stats import pearsonr

def analyze_correlation(df_success_param: pd.DataFrame):
    """
    Calculate and display statistical correlation between parameter count and success rate.
    
    Uses Pearson correlation coefficient to quantify the linear relationship.
    A coefficient close to +1 indicates strong positive correlation,
    close to -1 indicates strong negative correlation, and
    close to 0 indicates no linear correlation.
    
    Args:
        df_success_param: DataFrame containing parameter counts and success rates
    """
    # Calculate Pearson correlation coefficient and p-value
    correlation, p_value = pearsonr(df_success_param['Number_of_Parameters'], df_success_param['Success_Rate (%)'])
    
    # Print the results
    print(f"Pearson Correlation Coefficient between number of parameters and success rate: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Interpret the results
    if p_value < 0.05:
        significance = "statistically significant"
    else:
        significance = "not statistically significant"
        
    if abs(correlation) < 0.3:
        strength = "weak"
    elif abs(correlation) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
        
    print(f"This indicates a {strength} {significance} correlation between parameter count and success rate.")



def load_api_definitions(api_def_file: str) -> List[str]:
    """
    Load PyTorch API definitions from a text file.
    
    Each line in the file should contain a single PyTorch API signature.
    Empty lines are skipped.
    
    Args:
        api_def_file: Path to the file containing API definitions
        
    Returns:
        List of API signatures
        
    Raises:
        FileNotFoundError: If the API definition file doesn't exist
    """
    if not os.path.exists(api_def_file):
        raise FileNotFoundError(f"API definition file '{api_def_file}' not found.")
    
    with open(api_def_file, 'r') as f:
        apis = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(apis)} APIs from '{api_def_file}'.")
    return apis

def get_pickle_files(results_dir: str) -> List[str]:
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory '{results_dir}' does not exist.")
    
    pickle_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) 
                   if f.startswith('result_') and f.endswith('.pkl')]
    print(f"Found {len(pickle_files)} result files in '{results_dir}'.")
    return pickle_files

def get_program_files(programs_dir: str) -> List[str]:
    if not os.path.exists(programs_dir):
        raise FileNotFoundError(f"Programs directory '{programs_dir}' does not exist.")
    
    program_files = [os.path.join(programs_dir, f) for f in os.listdir(programs_dir) 
                     if f.startswith('program_') and f.endswith('.pkl')]
    print(f"Found {len(program_files)} program files in '{programs_dir}'.")
    return program_files

def get_parameter_files(parameters_dir: str) -> List[str]:
    if not os.path.exists(parameters_dir):
        raise FileNotFoundError(f"Parameters directory '{parameters_dir}' does not exist.")
    
    parameter_files = [os.path.join(parameters_dir, f) for f in os.listdir(parameters_dir) 
                       if f.startswith('fuzz_test_parameters_') and f.endswith('.csv')]
    print(f"Found {len(parameter_files)} parameter files in '{parameters_dir}'.")
    return parameter_files

def load_program_data(program_files: List[str]) -> Dict[int, Dict[str, Any]]:
    program_data = {}
    for file_path in tqdm(program_files, desc="Loading program files"):
        try:
            with open(file_path, 'rb') as f:
                program = pickle.load(f)
            filename = os.path.basename(file_path)
            program_id = int(filename.replace('program_', '').replace('.pkl', ''))
            if isinstance(program, dict):
                code = program.get('code', '')
                num_of_parameters = program.get('num_of_parameters', None)
            else:
                code = program
                num_of_parameters = None
            program_data[program_id] = {
                'code': code,
                'num_of_parameters': num_of_parameters
            }
        except Exception as e:
            print(f"Error loading '{file_path}': {e}")
    return program_data

def load_parameter_data(parameter_files: List[str]) -> Dict[int, List[List[str]]]:
    parameter_data = {}
    for file_path in tqdm(parameter_files, desc="Loading parameter files"):
        try:
            filename = os.path.basename(file_path)
            parts = filename.replace('fuzz_test_parameters_', '').replace('.csv', '').split('_')
            program_id = int(parts[0])
            
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                params = list(reader)
            parameter_data[program_id] = params
        except Exception as e:
            print(f"Error loading '{file_path}': {e}")
    return parameter_data

def load_results_stream(pickle_files: List[str]) -> Generator[Dict[str, Any], None, None]:
    for file_path in tqdm(pickle_files, desc="Streaming result files"):
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
            for result in results:
                yield result
        except Exception as e:
            print(f"Error loading '{file_path}': {e}")

def initialize_aggregates(apis: List[str]) -> Dict[str, Dict[str, float]]:
    aggregates = {}
    for api in apis:
        aggregates[api] = {
            'Total_Runs': 0,
            'Success': 0,
            'Mismatch': 0,
            'Errors': 0
        }
    return aggregates

def map_program_id_to_api(program_id: int, num_apis: int) -> int:
    return (program_id - 1) % num_apis

def categorize_error(error_message: str) -> str:
    """
    Categorize error messages into standardized error types.
    
    This function analyzes error messages from test runs and maps them to 
    common error categories for easier analysis and visualization.
    
    Args:
        error_message: The error message string from a test run
        
    Returns:
        Standardized error category name
    """
    if not error_message:
        return 'No Error'
        
    # Map common Python and PyTorch errors to categories
    if 'RuntimeError' in error_message:
        return 'RuntimeError'
    elif 'TypeError' in error_message:
        return 'TypeError'
    elif 'ValueError' in error_message:
        return 'ValueError'
    elif 'Timeout' in error_message:
        return 'Timeout'
    elif 'CPU time limit exceeded' in error_message:
        return 'Timeout'
    elif 'SyntaxError' in error_message:
        return 'SyntaxError'
    elif 'AttributeError' in error_message:
        return 'AttributeError'
    elif 'NameError' in error_message:
        return 'NameError'
    else:    
        return 'Other Errors'

def process_results_stream(results_stream: Generator[Dict[str, Any], None, None], aggregates: Dict[str, Dict[str, float]], apis: List[str], num_apis: int, error_categories: Dict[str, Dict[str, int]]):
    """
    Process the stream of test results to calculate aggregate statistics.
    
    This function iterates through all test results, maps each result to the 
    corresponding API, and updates the aggregate counters for success, mismatch,
    and error rates. It also categorizes errors by type for further analysis.
    
    Args:
        results_stream: Generator yielding test result dictionaries
        aggregates: Dictionary to store aggregated statistics per API
        apis: List of API signatures to map results to
        num_apis: Total number of APIs being tested
        error_categories: Dictionary to track error types per API
    """
    # Process each result in the stream
    for result in tqdm(results_stream, desc="Processing results"):
        program_id = result.get('program_id')
        if program_id is None:
            continue
        
        # Map the program ID to the corresponding API
        api_index = map_program_id_to_api(program_id, num_apis)
        if api_index >= len(apis):
            api_signature = 'Unknown API'
        else:
            api_signature = apis[api_index]
        
        # Update total run count for this API
        aggregates[api_signature]['Total_Runs'] += 1
        result_status = result.get('result')
        error = result.get('error')
        
        # Categorize the result and update appropriate counters
        if error:
            # Handle error case
            aggregates[api_signature]['Errors'] += 1
            category = categorize_error(error)
            if category in error_categories[api_signature]:
                error_categories[api_signature][category] += 1
            else:
                error_categories[api_signature][category] = 1
        elif result_status == 'CPU and GPU outputs are equal':
            # Handle success case
            aggregates[api_signature]['Success'] += 1
        elif result_status == 'CPU and GPU outputs are not equal':
            # Handle mismatch case
            aggregates[api_signature]['Mismatch'] += 1
        else:
            # Handle unexpected result
            pass


def calculate_rates(aggregates: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Calculate success, mismatch, and error rates for each API.
    
    Converts raw counts to percentage rates and constructs a DataFrame
    with both the raw counts and the computed rates for each API.
    
    Args:
        aggregates: Dictionary mapping APIs to their aggregated statistics
        
    Returns:
        DataFrame containing counts and rates for each API
    """
    data = []
    for api, stats in aggregates.items():
        # Get total number of runs for this API
        total = stats['Total_Runs']
        
        # Calculate rates, handling the zero division case
        if total == 0:
            success_rate = mismatch_rate = error_rate = 0.0
        else:
            success_rate = (stats['Success'] / total) * 100
            mismatch_rate = (stats['Mismatch'] / total) * 100
            error_rate = (stats['Errors'] / total) * 100
        
        # Build the record for this API
        data.append({
            'API': api,
            'Total_Runs': total,
            'Success': stats['Success'],
            'Mismatch': stats['Mismatch'],
            'Errors': stats['Errors'],
            'Success_Rate (%)': round(success_rate, 2),
            'Mismatch_Rate (%)': round(mismatch_rate, 2),
            'Error_Rate (%)': round(error_rate, 2)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def save_statistics_to_csv(df: pd.DataFrame, output_csv: str):
    """
    Save aggregated statistics to a CSV file.
    
    Args:
        df: DataFrame containing the statistics to save
        output_csv: Path to save the CSV file
    """
    df.to_csv(output_csv, index=False)
    print(f"Saved aggregated statistics to '{output_csv}'.")

def integrate_program_and_parameter_data(df: pd.DataFrame, program_data: Dict[int, Dict[str, Any]], parameter_data: Dict[int, List[List[str]]], num_apis: int, apis: List[str]) -> pd.DataFrame:
    """
    Integrate program and parameter data with API statistics.
    
    Calculates and adds two important metrics to the DataFrame:
    1. Average number of parameters required by each API
    2. Average number of parameter sets generated for each API
    
    Args:
        df: DataFrame containing API statistics
        program_data: Dictionary mapping program IDs to their metadata
        parameter_data: Dictionary mapping program IDs to parameter sets
        num_apis: Total number of APIs being tested
        apis: List of API signatures
        
    Returns:
        DataFrame with added parameter statistics columns
    """
    avg_num_parameters = []
    avg_params_per_set = []
    
    # Process each API
    for index, row in df.iterrows():
        api = row['API']
        
        # Find the API index in the list of APIs
        try:
            api_index = apis.index(api)
        except ValueError:
            api_index = -1
        
        # Handle unknown API
        if api_index == -1:
            avg_num_parameters.append(0)
            avg_params_per_set.append(0)
            continue
        
        # Find all program IDs that correspond to this API
        program_ids = [pid for pid in program_data.keys() if map_program_id_to_api(pid, num_apis) == api_index]
        
        # Calculate average number of parameters for this API
        num_params = [program_data[pid]['num_of_parameters'] for pid in program_ids 
                      if program_data[pid]['num_of_parameters'] is not None]
        avg_num = round(sum(num_params)/len(num_params), 2) if num_params else None
        avg_num_parameters.append(avg_num)
        
        # Calculate average parameters per set for this API
        params_per_set = [len(parameter_data[pid]) for pid in program_ids if pid in parameter_data]
        avg_params = round(sum(params_per_set)/len(params_per_set), 2) if params_per_set else None
        avg_params_per_set.append(avg_params)
    
    # Add the calculated averages to the DataFrame
    df['Average_Number_of_Parameters'] = avg_num_parameters
    df['Average_Parameters_Per_Set'] = avg_params_per_set
    
    return df

def generate_summary_visualizations(stats: pd.DataFrame, output_dir: str):
    """
    Generate a comprehensive set of visualizations summarizing test results.
    
    Creates multiple plots to visualize:
    - Overall test outcomes (success, mismatch, error)
    - Top APIs with highest mismatch and error rates
    - Distribution of success, mismatch, and error rates
    - Parameter count statistics per API
    - Correlations between parameters and error/mismatch rates
    
    Args:
        stats: DataFrame containing aggregated test statistics
        output_dir: Directory to save the generated visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    overall = {
        'Category': ['Success', 'Mismatch', 'Errors'],
        'Count': [stats['Success'].sum(), stats['Mismatch'].sum(), stats['Errors'].sum()]
    }
    overall_df = pd.DataFrame(overall)
    
    plt.figure(figsize=(8,6))
    sns.barplot(x='Category', y='Count', data=overall_df, palette=['green', 'orange', 'red'])
    plt.title('Overall Test Results')
    plt.ylabel('Number of Tests')
    plt.xlabel('Result Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_test_results.png'))
    plt.close()

    total_tests = overall_df['Count'].sum()
    for index, row in overall_df.iterrows():
        category = row['Category']
        count = row['Count']
        percentage = (count / total_tests) * 100
        print(f"{category}: {count} tests ({percentage:.2f}%)")
    
    top_mismatch = stats.sort_values(by='Mismatch_Rate (%)', ascending=False).head(nHead)
    top_mismatch['API'] = top_mismatch['API'].str.split('(').str[0]

    plt.figure(figsize=(15,8))
    sns.barplot(x='Mismatch_Rate (%)', y='API', data=top_mismatch, palette='Oranges_d')
    plt.title('Top 20 APIs with Highest Mismatch Rates')
    plt.xlabel('Mismatch Rate (%)')
    plt.ylabel('API')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top10_mismatch_rates.png'))
    plt.close()
    
    top_errors = stats.sort_values(by='Error_Rate (%)', ascending=False).head(nHead)
    top_errors['API'] = top_errors['API'].str.split('(').str[0]
    plt.figure(figsize=(15,8))
    sns.barplot(x='Error_Rate (%)', y='API', data=top_errors, palette='Reds_d')
    plt.title('Top 20 APIs with Highest Error Rates')
    plt.xlabel('Error Rate (%)')
    plt.ylabel('API')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top10_error_rates.png'))
    plt.close()
    
    plt.figure(figsize=(10,6))
    sns.histplot(stats['Success_Rate (%)'], bins=50, kde=True, color='green')
    plt.title('Distribution of Success Rates Across APIs')
    plt.xlabel('Success Rate (%)')
    plt.ylabel('Number of APIs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(10,6))
    sns.histplot(stats['Mismatch_Rate (%)'], bins=50, kde=True, color='orange')
    plt.yscale('log')
    plt.title('Distribution of Mismatch Rates Across APIs')
    plt.xlabel('Mismatch Rate (%)')
    plt.ylabel('Number of APIs')
    plt.tight_layout()
    plt.ylim(1e-1, 1e4)
    plt.savefig(os.path.join(output_dir, 'mismatch_rate_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(10,6))
    sns.histplot(stats['Error_Rate (%)'], bins=50, kde=True, color='red')
    plt.title('Distribution of Error Rates Across APIs')
    plt.xlabel('Error Rate (%)')
    plt.ylabel('Number of APIs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_rate_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(12,8))
    sns.barplot(x='API', y='Average_Number_of_Parameters', data=stats.sort_values('Average_Number_of_Parameters', ascending=False).head(nHead), palette='Blues_d')
    plt.title('Average Number of Parameters per API')
    plt.xlabel('API')
    plt.ylabel('Average Number of Parameters')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_parameters_per_api.png'))
    plt.close()
    
    plt.figure(figsize=(12,8))
    sns.barplot(x='API', y='Average_Parameters_Per_Set', data=stats.sort_values('Average_Parameters_Per_Set', ascending=False).head(nHead), palette='Purples_d')
    plt.title('Average Parameters Per Set per API')
    plt.xlabel('API')
    plt.ylabel('Average Parameters Per Set')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_params_per_set_per_api.png'))
    plt.close()
    
    try:
        x_bins = np.histogram_bin_edges(stats['Average_Number_of_Parameters'], bins='auto')
        y_bins = np.histogram_bin_edges(stats['Error_Rate (%)'], bins='auto')

        plt.figure(figsize=(10, 6))
        heatmap, xedges, yedges = np.histogram2d(
            stats['Average_Number_of_Parameters'], stats['Error_Rate (%)'], bins=[x_bins, y_bins]
        )

        sns.heatmap(
            heatmap.T, cmap="viridis", cbar=True,
            xticklabels=np.round(xedges[::5], 1), yticklabels=np.round(yedges[::5], 1)
        )
        plt.title('Number of Parameters vs. Error Rate (Heatmap)', fontsize=16, pad=20)
        plt.xlabel('Average Number of Parameters', fontsize=14)
        plt.ylabel('Error Rate (%)', fontsize=14)
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'params_vs_error_rate_heatmap.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating heatmap: {e}")
    
    try:
        x_bins = np.histogram_bin_edges(stats['Average_Parameters_Per_Set'], bins='auto')
        y_bins = np.histogram_bin_edges(stats['Mismatch_Rate (%)'], bins='auto')

        plt.figure(figsize=(10, 6))
        heatmap, xedges, yedges = np.histogram2d(
            stats['Average_Parameters_Per_Set'], stats['Mismatch_Rate (%)'], bins=[x_bins, y_bins]
        )

        sns.heatmap(
            heatmap.T, cmap="magma", cbar=True,
            xticklabels=np.round(xedges[::5], 1), yticklabels=np.round(yedges[::5], 1)
        )
        plt.title('Parameters Per Set vs. Mismatch Rate (Heatmap)', fontsize=16, pad=20)
        plt.xlabel('Average Parameters Per Set', fontsize=14)
        plt.ylabel('Mismatch Rate (%)', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        plt.savefig(os.path.join(output_dir, 'params_per_set_vs_mismatch_rate_heatmap.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating heatmap: {e}")
    print("Generated summary visualizations.")

def analyze_parameter_types(parameter_data: Dict[int, List[List[str]]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    all_params = []
    for params in parameter_data.values():
        for param_set in params:
            all_params.extend(param_set)
    
    df_params = pd.DataFrame(all_params, columns=['Parameter'])
    
    def infer_type(param):
        try:
            value = eval(param)
            return type(value).__name__
        except:
            return 'PyTorch Type'
    
    df_params['Type'] = df_params['Parameter'].apply(infer_type)
    
    plt.figure(figsize=(12,6))
    sns.countplot(y='Type', data=df_params, order=df_params['Type'].value_counts().index, palette='Set2')
    plt.title('Distribution of Parameter Types')
    plt.xlabel('Count')
    plt.ylabel('Parameter Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_types_distribution.png'))
    plt.close()
    
    type_counts = df_params['Type'].value_counts()
    type_counts.to_csv(os.path.join(output_dir, 'parameter_types_counts.csv'))
    
    print("Generated parameter types distribution analysis.")

def plot_parameters_distribution(program_data: Dict[int, Dict[str, Any]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    num_params = [pdata['num_of_parameters'] for pdata in program_data.values() if pdata['num_of_parameters'] is not None]
    
    plt.figure(figsize=(10,6))
    sns.histplot(num_params, bins=30, kde=True, color='blue')
    plt.title('Distribution of Number of Parameters per Program')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameters_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(10,6))
    sns.boxplot(x=num_params, color='cyan')
    plt.title('Boxplot of Number of Parameters per Program')
    plt.xlabel('Number of Parameters')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameters_boxplot.png'))
    plt.close()
    
    print("Generated parameters distribution visualizations.")

def generate_additional_insights(stats: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8,6))
    corr = stats[['Success_Rate (%)', 'Mismatch_Rate (%)', 'Error_Rate (%)']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Test Outcomes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    x_bins = np.histogram_bin_edges(stats['Success_Rate (%)'], bins='auto')
    y_bins = np.histogram_bin_edges(stats['Mismatch_Rate (%)'], bins='auto')

    plt.figure(figsize=(10, 6))
    heatmap, xedges, yedges = np.histogram2d(
        stats['Success_Rate (%)'], stats['Mismatch_Rate (%)'], bins=[x_bins, y_bins]
    )

    sns.heatmap(
        heatmap.T, cmap="viridis", cbar=True,
        xticklabels=np.round(xedges[::5], 1), yticklabels=np.round(yedges[::5], 1)
    )
    plt.title('Success Rate vs. Mismatch Rate (Heatmap)', fontsize=16, pad=20)
    plt.xlabel('Success Rate (%)', fontsize=14)
    plt.ylabel('Mismatch Rate (%)', fontsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'success_vs_mismatch_heatmap.png'))
    plt.close()
    try:
        x_bins = np.histogram_bin_edges(stats['Average_Number_of_Parameters'], bins='auto')
        y_bins = np.histogram_bin_edges(stats['Error_Rate (%)'], bins='auto')

        plt.figure(figsize=(10, 6))
        heatmap, xedges, yedges = np.histogram2d(
            stats['Average_Number_of_Parameters'], stats['Error_Rate (%)'], bins=[x_bins, y_bins]
        )
        sns.heatmap(
            heatmap.T, cmap="YlGnBu", cbar=True,
            xticklabels=np.round(xedges[::5], 1), yticklabels=np.round(yedges[::5], 1)
        )
        plt.title('Error Rate vs. Average Number of Parameters (Heatmap)')
        plt.xlabel('Average Number of Parameters')
        plt.ylabel('Error Rate (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_vs_parameters_heatmap.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating heatmap: {e}")
    

def initialize_error_categories(apis: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Initialize a nested dictionary to track error categories per API.
    
    Creates a structure to count occurrences of different error types
    for each API, making it easy to analyze which APIs are prone to
    which types of errors.
    
    Args:
        apis: List of API signatures
        
    Returns:
        Nested dictionary mapping APIs to error categories with zero counts
    """
    # Define the standard error categories to track
    categories = ['RuntimeError', 'TypeError', 'ValueError', 'Timeout', 
                  'SyntaxError', 'AttributeError', 'NameError', 'Other Errors']
    
    # Initialize the dictionary with zero counts for each category
    error_categories = {}
    for api in apis:
        error_categories[api] = {category: 0 for category in categories}
    
    return error_categories

def save_error_categories_to_csv(error_categories: Dict[str, Dict[str, int]], output_csv: str):

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['API', 'RuntimeError', 'TypeError', 'ValueError', 'Timeout', 'SyntaxError', 'AttributeError', 'NameError', 'Other Errors']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for api, categories in error_categories.items():
            row = {'API': api}
            row.update(categories)
            writer.writerow(row)
    print(f"Saved error categories to '{output_csv}'.")

def visualize_error_categories(error_categories: Dict[str, Dict[str, int]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    df_errors = pd.DataFrame.from_dict(error_categories, orient='index').reset_index()
    df_errors = df_errors.rename(columns={'index': 'API'})
    
    df_melted = df_errors.melt(id_vars='API', var_name='Error_Type', value_name='Count').head(nHead)
    
    plt.figure(figsize=(14,8))
    sns.barplot(x='API', y='Count', hue='Error_Type', data=df_melted, palette='Set3')
    plt.title('Error Categories per API')
    plt.xlabel('API')
    plt.ylabel('Number of Errors')
    plt.xticks(rotation=90)
    plt.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_categories_per_api.png'))
    plt.close()
    
    print("Generated error categories per API visualization.")

def visualize_parameter_types_vs_outcomes(program_data: Dict[int, Dict[str, Any]], results_stream: Generator[Dict[str, Any], None, None], output_dir: str):
    """
    Visualize the relationship between parameter count and test outcomes.
    
    Creates a bar chart showing how the distribution of test outcomes
    (success, mismatch, error) varies with the number of parameters.
    This helps identify whether APIs with more parameters tend to have
    higher failure rates.
    
    Args:
        program_data: Dictionary mapping program IDs to their metadata
        results_stream: Generator yielding test result dictionaries
        output_dir: Directory to save the generated visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process results to gather data for visualization
    data = []
    for result in tqdm(results_stream, desc="Processing for parameter types vs outcomes"):
        program_id = result.get('program_id')
        if program_id is None:
            continue
            
        # Get the number of parameters for this program
        num_params = program_data.get(program_id, {}).get('num_of_parameters')
        if num_params is None:
            continue
            
        # Categorize the outcome
        outcome = result.get('result')
        error = result.get('error')
        if error:
            outcome = 'Error'
        elif outcome == 'CPU and GPU outputs are equal':
            outcome = 'Success'
        elif outcome == 'CPU and GPU outputs are not equal':
            outcome = 'Mismatch'
        else:
            outcome = 'Unknown'
            
        # Add to dataset
        data.append({'Number_of_Parameters': num_params, 'Outcome': outcome})
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create visualization
    plt.figure(figsize=(10,6))
    sns.countplot(x='Number_of_Parameters', hue='Outcome', data=df, 
                  palette={'Success': 'green', 'Mismatch': 'orange', 'Error': 'red', 'Unknown': 'gray'})
    plt.title('Test Outcomes by Number of Parameters')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Count')
    plt.legend(title='Outcome')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outcomes_by_num_parameters.png'))
    plt.close()
    
    print("Generated outcomes by number of parameters visualization.")



def main():
    """
    Main execution function for the analysis module.
    
    This function orchestrates the entire analysis workflow:
    1. Loads API definitions and test results
    2. Processes result data and calculates statistics
    3. Integrates program and parameter metadata
    4. Generates various visualizations and insights
    5. Saves results to disk
    
    The analysis provides insights into:
    - Overall success/mismatch/error rates
    - Correlations between parameter counts and test outcomes
    - Distribution of error types across APIs
    - Patterns and trends in test results
    """
    # Step 1: Load API definitions
    try:
        apis = load_api_definitions(API_DEFINITION_FILE)
    except Exception as e:
        print(f"Failed to load API definitions: {e}")
        return
    
    num_apis = len(apis)
    
    # Step 2: Load result files
    try:
        pickle_files = get_pickle_files(RESULTS_DIR)
    except Exception as e:
        print(f"Failed to retrieve result pickle files: {e}")
        return
    
    if not pickle_files:
        print("No result pickle files found to process.")
        return
    
    # Step 3: Load program and parameter metadata
    try:
        program_files = get_program_files(PROGRAMS_DIR)
        parameter_files = get_parameter_files(PARAMETERS_DIR)
    except Exception as e:
        print(f"Failed to retrieve program or parameter files: {e}")
        return
    
    # Step 4: Process the data files
    program_data = load_program_data(program_files)
    parameter_data = load_parameter_data(parameter_files)
    
    # Step 5: Initialize data structures for analysis
    aggregates = initialize_aggregates(apis)
    error_categories = initialize_error_categories(apis)
    
    # Step 6: Process results and calculate statistics
    results_stream = load_results_stream(pickle_files)
    process_results_stream(results_stream, aggregates, apis, num_apis, error_categories)
    
    stats_df = calculate_rates(aggregates)
    
    # Step 7: Integrate program and parameter metadata
    stats_df = integrate_program_and_parameter_data(stats_df, program_data, parameter_data, num_apis, apis)
    
    # Step 8: Save statistics to CSV
    save_statistics_to_csv(stats_df, OUTPUT_CSV)
    
    # Step 9: Generate visualizations and insights
    generate_summary_visualizations(stats_df, OUTPUT_DIR)
    analyze_parameter_types(parameter_data, OUTPUT_DIR)
    plot_parameters_distribution(program_data, OUTPUT_DIR)
    generate_additional_insights(stats_df, OUTPUT_DIR)

    # Step 10: Analyze and visualize error patterns
    save_error_categories_to_csv(error_categories, os.path.join(OUTPUT_DIR, 'error_categories.csv'))
    visualize_error_categories(error_categories, OUTPUT_DIR)

    # Step 11: Analyze parameter count correlation with success rate
    results_stream = load_results_stream(pickle_files)
    df_success_param = calculate_success_by_parameter_count(program_data, results_stream)
    df_success_param.to_csv(os.path.join(OUTPUT_DIR, 'success_rate_vs_num_parameters.csv'), index=False)
    print("Success rates based on the number of parameters saved to 'success_rate_vs_num_parameters.csv'")
    print(df_success_param)

    # Step 12: Visualize and analyze correlations
    if df_success_param is not None and not df_success_param.empty:
        visualize_success_by_parameter_count(df_success_param, OUTPUT_DIR)
        analyze_correlation(df_success_param)

    # Step 13: Visualize outcome distributions by parameter count
    results_stream = load_results_stream(pickle_files)
    visualize_parameter_types_vs_outcomes(program_data, results_stream, OUTPUT_DIR)
    
if __name__ == "__main__":
    main()
