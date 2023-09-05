#!/bin/bash

# define the path to the script
script_path=$(dirname "$0")

# define the output directory
output_dir="$script_path/results"

# load configuration variables
source "$script_path/config/config.cfg"

data_file_path_p1=$(find "$script_path/data/" -name '*p1_test.txt' | sort | tr '\n' ' ')
data_file_path_p2=$(find "$script_path/data/" -name '*p1_test.txt' | sort | tr '\n' ' ')
model_path=$(find "$script_path/data/" -name '*.pth' | head -n 1)

if [[ -z "$data_file_path_p1" ]]; then
  echo "Error: no first segment of test data file found in data directory."
  exit 1
fi

if [[ -z "$data_file_path_p2" ]]; then
  echo "Error: no second segment of test data file found in data directory."
  exit 1
fi

if [[ -z "$n_labels" ]]; then
  echo "Error: number of unique labels should be specified in config file."
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

# fine-tune
evaluate_command="python3 $script_path/../../ehr_classification_with_bert \
          evaluate \
          --data-file-path $data_file_path_p1 $data_file_path_p2 \
          --model-path $model_path \
          --n-labels $n_labels \
          --unique-labels $unique_labels \
          --class-names $class_names \
          --results-path $output_dir "

if [[ -n "$batch_size" ]]; then
  evaluate_command+="--batch-size $batch_size "
fi

if [[ -n "$truncate_dataset_to" ]]; then
  evaluate_command+="--truncate-dataset-to $truncate_dataset_to "
fi

if [[ -n "$segmented" && "$segmented" == "true" ]]; then
  evaluate_command+="--segmented "
fi

if [[ -n "$unique_labels" ]]; then
  evaluate_command+="--unique-labels $unique_labels "
fi

if [[ -n "$class_names" ]]; then
  evaluate_command+="--class-names $class_names "
fi

echo "$evaluate_command"
eval "$evaluate_command"
