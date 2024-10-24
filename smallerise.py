import os
import pickle

def process_result_file(input_file, output_dir):
    # Read the input file
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    # Process each result entry
    processed_data = []
    for entry in data:
        # Create new entry with only necessary fields
        new_entry = {
            'program_id': entry['program_id'],
            'params' : entry['params'],
            'result': entry['result'],
            'output': entry['output'],
            'error': entry['error']
        }
        processed_data.append(new_entry)
    
    # Create output filename
    base_name = os.path.basename(input_file)
    output_file = os.path.join(output_dir, f"small_{base_name}")
    
    # Save processed data
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)

def main():
    # Create output directory if it doesn't exist
    output_dir = 'small_result_parts'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all result files
    input_dir = 'result_parts'
    for filename in os.listdir(input_dir):
        if filename.startswith('result_') and filename.endswith('.pkl'):
            input_file = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            process_result_file(input_file, output_dir)
            
    print("Processing complete!")

if __name__ == "__main__":
    main()