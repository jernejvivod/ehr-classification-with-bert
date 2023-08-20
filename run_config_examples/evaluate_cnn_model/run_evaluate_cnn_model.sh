#!/bin/bash

# define the path to the script
script_path=$(dirname "$0")

# define the output directory
output_dir="$script_path/results"

# load configuration variables
source "$script_path/config/config_evaluate_cnn_model.cfg"

test_data_file=$(find "$script_path/data/" -name '*test.txt' | head -n 1)
model_file=$(find "$script_path/results/" -name '*.pth' | head -n 1)

if [[ -z "$test_data_file" ]]; then
  echo "Error: no test data file found in data directory."
  exit 1
fi

if [[ -z "$unique_labels" ]]; then
  echo "Error: unique labels should be specified in config file."
  exit 1
fi

if [[ -z "$class_names" ]]; then
  echo "Error: class names should be specified in config file."
  exit 1
fi

# get evaluation results
evaluate_command="python3 $script_path/../../classification_with_embeddings \
          evaluate-cnn-model \
          --model-path $model_file \
          --test-data-path $test_data_file \
          --unique-labels $unique_labels \
          --class-names $class_names \
          --results-path $output_dir "

if [[ -n "$batch_size" ]]; then
  evaluate_command+="--batch-size $batch_size "
fi

echo "$evaluate_command"
eval "$evaluate_command"
