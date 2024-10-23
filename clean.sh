#!/bin/bash

# List of directories to clean
directories=(
    "result_parts"
    "program_files"
    "program_API_files"
    "temp_code"
    "parameter_API_files"
    "parameter_files"
)

# Iterate over each directory and remove all files
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "Removing files in $dir..."
        rm -f "$dir"/*
    else
        echo "Directory $dir does not exist. Skipping..."
    fi
done

echo "Cleanup completed!"
