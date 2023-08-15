#!/bin/bash

# define the path to the script
script_path=$(dirname "$0")

# define the output directory
output_dir="$script_path/results"

# load configuration variables
source "$script_path/config/config.cfg"

train_data_file=$(find "$script_path/data/" -name '*train.txt' | head -n 1)
test_data_file=$(find "$script_path/data/" -name '*test.txt' | head -n 1)
param_grid_path="$script_path/config/param_grid.json"

if [[ -z "$train_data_file" ]]; then
  echo "Error: no train data file found in data directory."
  exit 1
fi

if [[ -z "$test_data_file" ]]; then
  echo "Error: no test data file found in data directory."
  exit 1
fi

# get evaluation results
evaluate_command="python3 $script_path/../../classification_with_embeddings \
          evaluate-embeddings-model \
          --train-data-path $train_data_file \
          --test-data-path $test_data_file \
          --results-path $output_dir \
          --param-grid-path $param_grid_path \
          --no-grid-search "

if [[ -v method ]]; then
  evaluate_command+="--method $method "
fi

if [[ -v clf ]]; then
  evaluate_command+="--internal-clf $clf "

  case "$clf" in
  "logistic-regression" | "random-forest" | "svc")
    if [[ -v internal_clf_args ]]; then
      evaluate_command+="--internal-clf-args \"${internal_clf_args//,/ }\""
    fi
    ;;
  *)
    false
    ;;
  esac
fi


echo "$evaluate_command"
eval "$evaluate_command"
