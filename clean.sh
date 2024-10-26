#!/bin/bash

# Check if the user is sure about cleaning up
read -p "Are you sure you want to clean up all files? (yes/n): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Exiting..."
    exit 0
fi

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
