#!/bin/bash

# define the path to the script
script_path=$(dirname "$0")

# define the output directory
output_dir="$script_path/results"

# load configuration variables
source "$script_path/config/config_train_cnn_model.cfg"

train_data_file=$(find "$script_path/data/" -name '*train.txt' | head -n 1)

if [[ -z "$train_data_file" ]]; then
  echo "Error: no train data file found in data directory."
  exit 1
fi

# get evaluation results
train_val_split_command="python3 $script_path/../../classification_with_embeddings \
          train-test-split \
          --data-path $train_data_file \
          --output-dir $output_dir \
          --train-suffix train \
          --test-suffix val "

echo "$train_val_split_command"
eval "$train_val_split_command"
