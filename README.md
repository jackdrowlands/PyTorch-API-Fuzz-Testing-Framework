# PyTorch API Fuzz Testing Framework

Welcome to the **PyTorch API Fuzz Testing Framework**! This project leverages OpenAI's language models to automatically generate, execute, and analyze fuzz tests for various PyTorch APIs. The framework is designed to compare CPU and GPU responses to identify potential discrepancies and errors, ensuring the robustness and reliability of PyTorch operations.

## Project Overview

The PyTorch API Fuzz Testing Framework automates the generation and execution of test programs for PyTorch APIs. By utilizing OpenAI's language models, the framework creates diverse and comprehensive test cases that stress-test the APIs under various conditions. The results are then analyzed to identify inconsistencies between CPU and GPU executions, potential errors, and areas for improvement.

## Research Paper

This framework is documented in our paper: **"PyTorch API Fuzz Testing Framework: A Large Language Model-Based Approach for Testing Deep Learning Libraries"**. The research demonstrates how Large Language Models (LLMs) can be leveraged for systematic differential testing of deep learning libraries, focusing on the comparison between CPU and GPU implementations of PyTorch operations.

## Features

- **Automated Test Generation**: Generates Python test programs for specified PyTorch APIs using OpenAI's language models.
- **Fuzz Testing**: Creates diverse parameter sets to explore edge cases and potential failure points.
- **Differential Testing**: Systematically compares CPU and GPU implementations to identify inconsistencies.
- **Parallel Execution**: Executes multiple tests concurrently to optimize performance.
- **Result Caching**: Saves generated programs and parameters to avoid redundant computations.
- **Comprehensive Analysis**: Aggregates and visualizes test results, including success rates, error types, and performance metrics.
- **Resource Management**: Implements memory and time constraints to ensure efficient testing.
- **Error Categorization**: Classifies errors into categories like RuntimeError, TypeError, ValueError for better insights.

## Prerequisites

Before setting up the project, ensure you have the following:

- **Python 3.8 or higher**
- **PyTorch**: Compatible with your hardware (CPU/GPU)
- **OpenAI API Key**: For generating test programs and parameters
- **Git**: For version control
- **Internet Connection**: Required for API calls to OpenAI
- **350GB of Free Space**: Required to store all of the results

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Topics-Group-AI-For-Fuzzing-AI/topics-fuzzing-ai/tree/jack-structered-outputs.git
   cd topics-fuzzing-ai
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. **Set Up OpenAI API Key**

   The framework requires an OpenAI API key to generate test programs and parameters. Set the `OPENAI_API_KEY` environment variable:

     ```bash
     export OPENAI_API_KEY='your-openai-api-key'
     ```


2. **API Definitions**

   Ensure that the `api_def_torch.txt` file exists in the project root directory. This file should contain the list of PyTorch API signatures you wish to test, one per line. 

## Framework Architecture

The PyTorch API Fuzz Testing Framework employs a three-phase architecture:

1. **Test Program Generation**:
   - Uses LLMs (primarily GPT-4o-mini) to generate structured test programs for each PyTorch API
   - Test programs include tensor creation, API invocation, and comparison code
   - Programs are designed to be executable directly with parameter injection

2. **Parameter Generation**:
   - Leverages LLMs to create diverse parameter sets that test edge cases
   - Parameters are constrained to prevent memory overflow (under 4GB)
   - Ensures tensor dimensions stay reasonable (<1000)
   - Enforces type compatibility with the operations being tested

3. **Test Execution and Analysis**:
   - Executes each test program with various parameter sets
   - Runs operations on both CPU and GPU devices
   - Compares outputs using torch.allclose() with appropriate tolerance
   - Records successes, errors, and mismatches in structured formats
   - Produces comprehensive visualizations and statistical analyses

## Key Findings

Our extensive testing of PyTorch APIs revealed several important insights:

- **Success Rate**: Approximately 30% of tests ran successfully without errors or mismatches
- **Error Patterns**: Runtime errors were the dominant issue (~65% of tests), particularly with specialized mathematical functions
- **Mismatch Detection**: About 5% of tests revealed discrepancies between CPU and GPU implementations
- **Parameter Sensitivity**: Tests with 3-7 parameters showed higher success rates than those with very few or many parameters
- **Error Correlation**: Strong negative correlation (-0.93) between success rates and error rates

APIs with the highest mismatch rates were often related to random number generation and tensor operations, suggesting these areas may benefit from additional attention in PyTorch development.

## Usage

The project consists of several scripts, each handling different aspects of the testing and analysis process. Below is a step-by-step guide to running the framework.

### 1. Running Fuzz Tests

This is the primary step where test programs are generated and executed.

**Script**: `run.py`

**Description**: Generates or loads test programs for each specified PyTorch API, creates or loads fuzz test parameters, executes the tests, and caches the results.

**Command**:

```bash
python run.py
```

**Optional Arguments**:

- `num_programs`: Total number of test programs to generate (default: 1584)
- `num_apis`: Number of APIs to test per program (default: 1)
- `start_id`: Starting ID for test programs (useful for resuming) (default: 0)
- `max_workers`: Number of concurrent threads (default: 12)

**Note**: Ensure that your system has sufficient resources to handle the specified number of concurrent threads and test executions.

### 2. Calculating Token Usage

This step analyzes the number of tokens used in API calls to OpenAI.

**Script**: `calc_tokens.py`

**Description**: Extracts and summarizes token usage from the generated test programs and parameters.

**Command**:

```bash
python calc_tokens.py
```

**Output**:

- `usage_summary.csv`
- `program_summary.csv`

These CSV files contain detailed statistics about token usage across different test cases.

### 3. Analyzing Results

After running tests, it's crucial to analyze the outcomes to identify patterns, successes, and failures.

**Script**: `analyse.py`

**Description**: Processes the test results, aggregates statistics, and generates visualizations to provide insights into the performance and reliability of the tested PyTorch APIs.

**Command**:

```bash
python analyse.py
```

**Output**:

- `analysis_report.csv`: Aggregated statistics of test results.
- `analysis_outputs/`: Directory containing various visualizations

## api_def_torch.txt

This list was sourced from [TitanFuzz](https://github.com/ise-uiuc/TitanFuzz).

## Analysis Outputs

After running the analysis script, you'll find several visualization outputs in the `analysis_outputs` directory:

### Overall Statistics
- `overall_test_results.png`: Breakdown of test outcomes (success, mismatch, error)
- `success_rate_distribution.png`: Distribution of success rates across APIs
- `mismatch_rate_distribution.png`: Distribution of mismatch rates across APIs
- `error_rate_distribution.png`: Distribution of error rates across APIs

### API-Specific Analysis
- `top10_mismatch_rates.png`: APIs with the highest mismatch rates
- `top10_error_rates.png`: APIs with the highest error rates
- `error_categories_per_api.png`: Breakdown of error types per API

### Parameter Analysis
- `average_parameters_per_api.png`: Average number of parameters per API
- `parameter_types_distribution.png`: Distribution of parameter types used in tests
- `parameters_distribution.png`: Distribution of parameter counts across tests
- `success_rate_vs_num_parameters.png`: Relationship between parameter count and success rate

### Correlation Analysis
- `correlation_matrix.png`: Correlation between success, mismatch, and error rates
- `success_vs_mismatch_heatmap.png`: Heatmap showing relationship between success and mismatch rates
- `params_vs_error_rate_heatmap.png`: Heatmap showing relationship between parameter count and error rates

## Limitations and Future Work

While the framework provides valuable insights into PyTorch API behavior, some limitations include:

1. **Resource Constraints**: Tests run with a 4GB memory limit and 90-second timeout
2. **Parameter Coverage**: While diverse, parameter generation may not cover all possible edge cases
3. **Scope**: Currently focuses on numerical discrepancies without analyzing performance differences
4. **Test Complexity**: Most tests involve single API calls rather than complex interactions

Future work could include:
- Expanding testing to other deep learning frameworks
- Incorporating performance testing alongside correctness testing
- Implementing automatic bug reporting mechanisms
- Developing more sophisticated program generation capabilities for testing complex API interactions
