# PyTorch API Fuzz Testing Framework

Welcome to the **PyTorch API Fuzz Testing Framework**! This project leverages OpenAI's language models to automatically generate, execute, and analyze fuzz tests for various PyTorch APIs. The framework is designed to compare CPU and GPU responses to identify potential discrepancies and errors, ensuring the robustness and reliability of PyTorch operations.

## Project Overview

The PyTorch API Fuzz Testing Framework automates the generation and execution of test programs for PyTorch APIs. By utilizing OpenAI's language models, the framework creates diverse and comprehensive test cases that stress-test the APIs under various conditions. The results are then analyzed to identify inconsistencies between CPU and GPU executions, potential errors, and areas for improvement.

## Features

- **Automated Test Generation**: Generates Python test programs for specified PyTorch APIs using OpenAI's language models.
- **Fuzz Testing**: Creates diverse parameter sets to explore edge cases and potential failure points.
- **Parallel Execution**: Executes multiple tests concurrently to optimize performance.
- **Result Caching**: Saves generated programs and parameters to avoid redundant computations.
- **Comprehensive Analysis**: Aggregates and visualizes test results, including success rates, error types, and performance metrics.
- **Resource Management**: Implements memory and time constraints to ensure efficient testing.

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
