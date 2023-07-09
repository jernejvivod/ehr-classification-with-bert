#!/bin/bash

# define the path to the script
script_path=$(dirname "$0")

# define the output directory
output_dir="$script_path/results"

# load configuration variables
source "$script_path/config/config.cfg"

data_file_path=$(find "$script_path/data/" -name '*test.txt' | sort | tr '\n' ' ')
model_path=$(find "$script_path/data/" -name '*.pth' | head -n 1)

if [[ -z "$data_file_path" ]]; then
  echo "Error: no text data file found in data directory."
  exit 1
fi

# fine-tune
evaluate_command="python3 $script_path/../../ehr_classification_with_bert \
          evaluate \
          --data-file-path $data_file_path \
          --model-path $model_path \
          --results-path $output_dir "

if [[ -v n_labels ]]; then
  evaluate_command+="--n-labels $n_labels "
fi

if [[ -v batch_size ]]; then
  evaluate_command+="--batch-size $batch_size "
fi

if [[ -v truncate_dataset_to ]]; then
  evaluate_command+="--truncate-dataset-to $truncate_dataset_to "
fi

if [[ -v segmented && "$segmented" == "true" ]]; then
  evaluate_command+="--segmented "
fi

if [[ -v unique_labels ]]; then
  evaluate_command+="--unique-labels $unique_labels "
fi

if [[ -v class_names ]]; then
  evaluate_command+="--class-names $class_names "
fi

echo "$evaluate_command"
eval "$evaluate_command"
